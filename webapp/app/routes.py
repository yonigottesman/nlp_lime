from app import app
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.responses import PlainTextResponse, RedirectResponse,JSONResponse
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from app.models import tfidf_model

templates = Jinja2Templates(directory='app/templates')

@app.route('/',methods=['GET','POST'])
async def homepage(request):
    data = await request.form()
    if 'text' in data:
        text = data['text']
        exp = tfidf_model.get_lime_exp(text)
        return templates.TemplateResponse('index.html', {'request': request,
                                                         'text': text,
                                                         'exp':exp})
    else:
        return templates.TemplateResponse('index.html', {'request': request,'text': 'The food was good'})