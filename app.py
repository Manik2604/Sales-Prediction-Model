import os
import gradio as gr
import joblib
import pandas as pd


# ==============================
# Load Machine Learning Model
# ==============================

MODEL_PATH = "Sales_Prediction_Model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise Exception(
        f"Model file not found or cannot be loaded.\n"
        f"Make sure '{MODEL_PATH}' is in the same folder.\n\nError: {e}"
    )


# ==============================
# Prediction Function
# ==============================

def predict_sales(tv, radio, newspaper):

    try:
        input_data = pd.DataFrame(
            [[tv, radio, newspaper]],
            columns=["TV", "Radio", "Newspaper"]
        )

        result = model.predict(input_data)[0]

        return f"📈 Predicted Sales: {result:.2f} units"

    except Exception as e:
        return f"❌ Error: {str(e)}"



# ==============================
# Custom CSS
# ==============================

css = """

body{
    background:#eef2f7;
}

.gradio-container{
    max-width:1100px !important;
    margin:auto;
}


.box{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px #cccccc;
}


footer{
    display:none;
}

"""


# ==============================
# Gradio UI
# ==============================

with gr.Blocks(
    css=css,
    title="Sales Prediction App"
) as app:


    with gr.Column(elem_classes="box"):

        gr.Markdown(
            """
            # 📊 Sales Prediction using Machine Learning

            Predict product sales using advertisement budgets.
            """
        )


        gr.Markdown("---")


        with gr.Row():


            # Input Section

            with gr.Column():

                gr.Markdown(
                    "## 📥 Advertisement Budget"
                )


                tv = gr.Number(
                    label="TV Advertisement Budget",
                    value=100
                )


                radio = gr.Number(
                    label="Radio Advertisement Budget",
                    value=25
                )


                newspaper = gr.Number(
                    label="Newspaper Advertisement Budget",
                    value=20
                )


                predict = gr.Button(
                    "🚀 Predict Sales"
                )


                output = gr.Textbox(
                    label="Prediction"
                )



            # Information Section

            with gr.Column():

                gr.Markdown(
                    """
                    ## 👨‍💻 Developer

                    **Name:** Manik Jindal

                    **College:**  
                    Panipat Institute of Engineering and Technology


                    ## 🛠 Technology Used

                    - Python
                    - Pandas
                    - Scikit-Learn
                    - Decision Tree Regression
                    - Joblib
                    - Gradio


                    ## 📌 About Project

                    This application predicts product sales
                    based on advertisement spending on:

                    📺 TV  
                    📻 Radio  
                    📰 Newspaper

                    The prediction is generated using
                    a trained Decision Tree Regression model.
                    """
                )


        predict.click(
            fn=predict_sales,
            inputs=[
                tv,
                radio,
                newspaper
            ],
            outputs=output
        )



# ==============================
# Run Application
# ==============================

if __name__ == "__main__":

    app.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
