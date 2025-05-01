import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

writer = SummaryWriter('runs/retrieval_train')

class MoEGate(nn.Module):
    def __init__(self, moe_mode, embedding_size, hidden_size,  output_size):
        super().__init__()

        self.query_hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.memory_hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        if moe_mode == 'general':
            raise
        elif moe_mode == 'trainable':
            self.initialize_trainable_mode()
        else:
            raise "Mode Error!"
    
    def initialize_trainable_mode(self):
        self.initialize_trainable_layer(self.query_hidden_layer)
        self.initialize_trainable_layer(self.memory_hidden_layer)
        self.initialize_trainable_layer(self.output_layer)
    
    def initialize_trainable_layer(self, layer):
        torch.nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, query, memory):
        query_h = self.query_hidden_layer(query)
        memory_h = self.memory_hidden_layer(memory)

        h = torch.sigmoid(query_h + memory_h)
        return torch.softmax(self.output_layer(h),dim=1)

class ImportanceScorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_size = config['embedding_size']

        self.W_q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_m = nn.Linear(self.embedding_size, self.embedding_size)
    
    def forward(self, h_q, h_m):
        e_q = self.W_q(h_q)
        e_m = self.W_m(h_m)

        score = torch.cosine_similarity(e_q,e_m)
        return score

class EmotionScorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_category']

        self.hidden_layer = nn.Linear(self.embedding_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 8)
    
    def forward(self, x):
        h = torch.tanh(self.hidden_layer(x))

        score = self.output_layer(h)
        return score

class ScoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.embedding_size = config['embedding_size']
        self.time_rank = config['time_rank']
        self.output_size = self.time_rank + 3
        self.moe_gate = MoEGate(config['moe_mode'], config['embedding_size'], config['hidden_size'], self.output_size)
        self.moe_gate.cuda()

        self.initialize_metrics(config['metrics_path'])
    
    def initialize_metrics(self, metrics_path):
        if metrics_path == False:
            raise
        else:
            importance_param_path = os.path.join(metrics_path, 'importance_score.pickle')
            self.importance_scorer = ImportanceScorer({
                'embedding_size': self.embedding_size
            })

            self.importance_scorer.load_state_dict(torch.load(importance_param_path).state_dict())
            self.importance_scorer = self.importance_scorer.cuda()
            
            emotion_param_path = os.path.join(metrics_path,'emotion_score.pickle')
            self.emotion_scorer = EmotionScorer({
                'embedding_size': self.embedding_size,
                'hidden_size': 256,
                'output_category': 8,
            })
            self.emotion_scorer.load_state_dict(torch.load(emotion_param_path).state_dict())
            self.emotion_scorer = self.emotion_scorer.cuda()

    def calculate_recency_scores(self, delta_time):
        recency_scores = torch.ones(delta_time.size()).unsqueeze(dim=0).cuda()
        for i in range(self.time_rank-1):
            recency_scores = torch.cat((recency_scores, (recency_scores[-1] * delta_time).unsqueeze(dim=0)), dim=0)
        return recency_scores.t()
    
    def calculate_emotion_scores(self, query, memory):
        # print(query.shape, memory.shape)
        query_emo = self.emotion_scorer(query)
        # print(query_emo.shape)
        query_emo_norm = torch.norm(query_emo, p=2, dim=1)
        # print(query_emo_norm.shape)
        memory_emo = self.emotion_scorer(memory)
        # print(memory_emo.shape)
        memory_emo_norm = torch.norm(memory_emo, p=2, dim=1)
        # print(memory_emo_norm.shape)
        emotion_scores = torch.sum(torch.mul(memory_emo, query_emo), dim=1) / query_emo_norm / memory_emo_norm

        return emotion_scores
    
    def calculate_importance_scores(self, query, memory):
        original_importance_score = self.importance_scorer(query, memory)
        scaled_importance_score = torch.sigmoid(original_importance_score)
        return scaled_importance_score
    
    def forward(self, query, memory, memory_delta_time):
        moe_gate_score = self.moe_gate(query, memory)
        # print('MoE Gate Score Shape:', moe_gate_score.shape)

        # print('Memory Text Shape:', memory.shape, 'Memory Time Shape:', query.shape)
        semantic_score = torch.sum(torch.mul(memory, query), dim=1)
        # print('Semantic Score Shape:', semantic_score.shape)

        recency_scores = - 1 * self.calculate_recency_scores(memory_delta_time)
        # print('Recency Score Shape:', recency_scores.shape)

        emotion_score = self.calculate_emotion_scores(query, memory)
        # print('Emotion Score Shape:', emotion_score.shape)

        importance_score = self.calculate_importance_scores(query, memory)
        # print('Importance Score Shape:', importance_score.shape)

        scores = torch.cat((semantic_score.unsqueeze(dim=1), recency_scores, emotion_score.unsqueeze(dim=1), importance_score.unsqueeze(dim=1)),dim=1)
        # print('Score Shape:', scores.shape)

        combined_score = torch.mul(moe_gate_score, scores).sum(dim=1)
        # print('Combine Score Shape:', combined_score.shape)

        return combined_score

model_config = {
    'moe_mode': 'trainable',
    'metrics_path': '[Path]',
    'embedding_size': 768,
    'hidden_size': 256,
    'time_rank': 5,
}

train_config = {
    'batch_size': 256,
    'total_epoch': 150,
    'lr': 0.01
}

def retrieval_offline_train():
    dataset_path = '[Path for retrieval_tensor.pickle]'
    moe_gate_save_path = '[Path for trained_moe_gate.pickle]'
    dataset = torch.load(dataset_path)
    query_text_tensor, core_text_tensor, inverse_text_tenosr, core_time_tensor, inverse_time_tensor, coef_tensor = dataset

    train_num = int(query_text_tensor.shape[0] * 0.9)
    # test_num = query_text_tensor.shape[0] - train_num

    train_dataset = TensorDataset(
        query_text_tensor[:train_num].cuda(), core_text_tensor[:train_num].cuda(), core_time_tensor[:train_num].cuda(),
        inverse_text_tenosr[:train_num].cuda(), inverse_time_tensor[:train_num].cuda(), coef_tensor[:train_num].cuda()
    )
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)

    test_dataset = TensorDataset(
        query_text_tensor[train_num:].cuda(), core_text_tensor[train_num:].cuda(), core_time_tensor[train_num:].cuda(),
        inverse_text_tenosr[train_num:].cuda(), inverse_time_tensor[train_num:].cuda(), coef_tensor[train_num:].cuda()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    score_model = ScoreModel(model_config)
    score_model = score_model.cuda()
    optimizer = optim.SGD(score_model.parameters(), lr=train_config['lr'])

    for epoch in range(train_config['total_epoch']):
        train_loss = torch.tensor(0.0)
        for batch_id, (batch_query, batch_core_text, batch_core_time, batch_inverse_text, batch_inverse_time, batch_coef) in enumerate(train_dataloader):
            optimizer.zero_grad()
            core_score = score_model(batch_query, batch_core_text, batch_core_time)

            inverse_score = score_model(batch_query, batch_inverse_text, batch_inverse_time)

            loss = - torch.sigmoid(core_score - inverse_score) + torch.sigmoid(inverse_score - core_score)

            scaled_loss = torch.mul(loss, batch_coef)
            avg_loss = torch.mean(scaled_loss)

            avg_loss.backward()
            optimizer.step()

            train_loss += avg_loss.detach().cpu()
            
        train_loss /= (batch_id + 1)
        print('(Train) Epoch %d loss: %f' % (epoch, train_loss))

        valid_loss = torch.tensor(0.0)
        for batch_id, (batch_query, batch_core_text, batch_core_time, batch_inverse_text, batch_inverse_time, batch_coef) in enumerate(test_dataloader):
            core_score = score_model(batch_query, batch_core_text, batch_core_time)
            inverse_score = score_model(batch_query, batch_inverse_text, batch_inverse_time)
            loss = - torch.sigmoid(core_score - inverse_score) + torch.sigmoid(inverse_score - core_score)
            scaled_loss = torch.mul(loss, batch_coef)
            avg_loss = torch.mean(scaled_loss)
            
            valid_loss += avg_loss.detach().cpu()
        valid_loss /= (batch_id + 1)
        print('------ (Valid) Epoch %d loss: %f' % (epoch, valid_loss))

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

    torch.save(score_model.moe_gate, moe_gate_save_path)



if __name__ == '__main__':
    retrieval_offline_train()
    writer.close()