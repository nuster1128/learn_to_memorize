from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from memengine import *

class Data(BaseModel):
    session_id: str
    operation: str
    kwargs: dict

OPERATIONS = ['reset', 'store', 'recall', 'manage', 'optimize', 'display']
KWARG_OPERATIONS = ['reset', 'manage', 'optimize', 'display']

memengine_server = FastAPI()

service_database = {}

@memengine_server.get("/")
async def check_connection():
    return "Successful connection."

@memengine_server.post("/init/")
async def create_new_session():
    session_id = str(uuid.uuid4())
    
    service_database[session_id] = None
    print('Create a new session: %s' % session_id)
    return {'info': 'Successully initilize your client!', 'session_id': session_id}

@memengine_server.post("/operation/")
async def memory_operation(data: Data):
    print(data.session_id)
    if data.session_id not in service_database:
        return {'info': 'Session_id %s is not found in the server. You may restart the client to create a new session.'}
    memory = service_database[data.session_id]
    if data.operation == 'initilize':
        try:
            service_database[data.session_id] = eval('%s' % data.kwargs['method'])(MemoryConfig(data.kwargs['config']))
            return {'info': 'Successully initilize the memory in server.'}
        except Exception as e:
            print('Fail to excuate due to: %s' % e)
            return {'info': 'Fail to excuate due to: %s' % e}
    elif data.operation in OPERATIONS:
        try:
            if data.operation in KWARG_OPERATIONS:
                res = getattr(memory, data.operation)(**data.kwargs)
            else:
                res = getattr(memory, data.operation)(data.kwargs)
            return {'info': 'Successfully excuate %s.' % data.operation, 'response': res}
        except Exception as e:
            print('Fail to excuate due to: %s' % e)
            return {'info': 'Fail to excuate due to: %s' % e}
    else:
        print('Operation not found.')
        return {'info': 'Operation not found.'}