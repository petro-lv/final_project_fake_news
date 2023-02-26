from newspaper import Article
import newspaper


def pipeline(preprocessor, model):
    def inner(raw_text):
        processed_text = preprocessor(raw_text)
        pred = model(processed_text)
        result = {k: v for k, v in zip(["Propaganda", "Not Propaganda"], [pred, 1 - pred])}
        return result
    return inner


def parse_title(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    return article.title


def article_predict(url, pipeline):
    try:
        title = parse_title(url)
    except:
        return {}, "Couldn't Parse"
    else:
        return pipeline(title), title
    

def message_predict(text, pipeline):
    return pipeline(text)