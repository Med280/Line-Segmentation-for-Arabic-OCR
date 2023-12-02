from fastapi import FastAPI
import uvicorn
from configuration.config import server_config
from source.apis import router

app = FastAPI()
app.include_router(router)
if __name__ == '__main__':
    uvicorn.run(app=app, host=server_config.HOST, port=server_config.PORT)
