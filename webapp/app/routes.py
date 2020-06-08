from app import app
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.responses import PlainTextResponse, RedirectResponse,JSONResponse
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from app.models import tfidf_model, fasttext_model

templates = Jinja2Templates(directory='app/templates')

@app.route('/',methods=['GET','POST'])
async def homepage(request):
    data = await request.form()
    if 'text' in data:
        text = data['text']
        tfidf_exp = tfidf_model.get_lime_exp(text)
        fasttext_exp = fasttext_model.get_lime_exp(text)
        
        return templates.TemplateResponse('index.html', {'request': request,
                                                         'text': text,
                                                         'results':[{'title':'tfidf','exp':tfidf_exp}, {'title':'fasttext','exp':fasttext_exp}]})
    else:
        return templates.TemplateResponse('index.html', {'request': request,'text': 'The food was good'})