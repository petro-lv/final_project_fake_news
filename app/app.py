import gradio as gr
from functools import partial
from helper import pipeline, message_predict, article_predict
from preprocessors import preprocessor_svm, preprocessor_bert
from models import svm_model, bert_model


svm_pipeline = pipeline(preprocessor_svm, svm_model)
bert_pipeline = pipeline(preprocessor_bert, bert_model)

message_predict_svm = partial(message_predict, pipeline=svm_pipeline)
message_predict_bert = partial(message_predict, pipeline=bert_pipeline)

article_predict_svm = partial(article_predict, pipeline=svm_pipeline)
article_predict_bert = partial(article_predict, pipeline=bert_pipeline)


article_examples = [
    "https://news.liga.net/politics/news/budanov-prognoziruet-reshayuschuyu-bitvu-mejdu-ukrainoy-i-rf-etoy-vesnoy",
    "https://glavcom.ua/ru/news/istrebiteli-dlja-vsu-reznikov-anonsiroval-khoroshie-novosti-912067.html"
]


message_examples = [
    "Пока украинские националисты призывают к ведению интернет-войны с помощью фотографий подбитой техники ВС РФ, в сети появляются все больше снимков сожженной техники ВСУ",
    "На Бориспольской трассе ВСУ пытаются задержать русских оригинальным способом. Публикует украинское информагентство УНИАН"
]


with gr.Blocks() as demo:
    gr.Markdown("""
    # Fake News Detection Demo
    """)
    with gr.Tab("URL"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    url = gr.Text(label="Enter a URL")
                with gr.Row():
                    btn_svm_url = gr.Button("SVM")
                    btn_bert_url = gr.Button("BERT")
                with gr.Row():
                    gr.Examples(article_examples, inputs=[url])

            with gr.Column():
                pred_url = gr.Label()
                title_url = gr.Text(label="Parsed Title")
            btn_svm_url.click(article_predict_svm, inputs=[url, ], outputs=[pred_url, title_url])
            btn_bert_url.click(article_predict_bert, inputs=[url, ], outputs=[pred_url, title_url])
            
    with gr.Tab("Message"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    message = gr.Text(label="Enter a Message")
                with gr.Row():
                    btn_svm_message = gr.Button("SVM")
                    btn_bert_message = gr.Button("BERT")
                with gr.Row():
                    gr.Examples(message_examples, inputs=[message])

            with gr.Column():
                pred_message = gr.Label()
            btn_svm_message.click(message_predict_svm, inputs=[message, ], outputs=[pred_message])
            btn_bert_message.click(message_predict_bert, inputs=[message, ], outputs=[pred_message])
        
        
print('Starting Server')
demo.launch(server_name="0.0.0.0", server_port=8080, share=True)

