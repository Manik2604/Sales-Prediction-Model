import os
import gradio as gr
import joblib
import pandas as pd


# =====================================
# Load Trained Model
# =====================================

MODEL_FILE = "Sales_Prediction_Model.pkl"

try:
    model = joblib.load(MODEL_FILE)

    # Automatically detect training feature order
    try:
        FEATURES = list(model.feature_names_in_)
    except:
        FEATURES = ["TV", "Radio", "Newspaper"]

except Exception as e:
    raise Exception(
        f"Model loading failed.\n"
        f"Keep {MODEL_FILE} in the same folder.\n\n{e}"
    )


print("Model Features:", FEATURES)



# =====================================
# Prediction Function
# =====================================

def predict_sales(tv, radio, newspaper):

    try:

        input_values = {
            "TV": tv,
            "Radio": radio,
            "Newspaper": newspaper
        }


        # Arrange input according to model training order
        input_data = pd.DataFrame(
            [[input_values[col] for col in FEATURES]],
            columns=FEATURES
        )


        prediction = model.predict(input_data)[0]


        return f"📈 Predicted Sales: {prediction:.2f} units"


    except Exception as e:

        return f"❌ Error: {str(e)}"



# =====================================
# Custom CSS
# =====================================

custom_css = """

.gradio-container{

    background:#eaf2f8;
}


.main-box{

    background:black;
    border-radius:20px;
    padding:30px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.15);

}


.main-box *{

    color:white !important;

}


button{

    font-weight:bold !important;

}


footer{

    display:none;

}

"""



# =====================================
# Gradio Application
# =====================================

with gr.Blocks(
    css=custom_css,
    title="Sales Prediction using Decision Tree"
) as app:


    with gr.Column(
        elem_classes="main-box"
    ):


        gr.Markdown(
            """
            # 📊 Sales Prediction using Machine Learning

            ### Predict Product Sales based on Advertisement Budget
            """
        )


        gr.Markdown("---")



        with gr.Row():


            # ==========================
            # Input Section
            # ==========================

            with gr.Column(scale=2):


                gr.Markdown(
                    "## 📥 Enter Advertisement Budget"
                )


                tv = gr.Number(
                    label="📺 TV Advertisement Budget",
                    value=100
                )


                radio = gr.Number(
                    label="📻 Radio Advertisement Budget",
                    value=25
                )


                newspaper = gr.Number(
                    label="📰 Newspaper Advertisement Budget",
                    value=20
                )


                predict_btn = gr.Button(
                    "🚀 Predict Sales"
                )


                output = gr.Textbox(
                    label="Prediction Result"
                )



            # ==========================
            # Developer Details
            # ==========================

            with gr.Column(scale=1):


                gr.Markdown(
                    """
                    ## 👨‍💻 Developer Details


                    **Name:**  
                    Manik Jindal


                    **College:**  
                    Panipat Institute of Engineering and Technology


                    **Project Title:**  
                    Sales Prediction using Decision Tree Regression


                    ## 🛠 Technology Stack


                    - Python
                    - Pandas
                    - Scikit-Learn
                    - Decision Tree Regressor
                    - Joblib
                    - Gradio


                    ## 📌 About Project


                    This application predicts expected product sales
                    using advertisement spending data.


                    The model considers:


                    📺 TV Advertisement  
                    📻 Radio Advertisement  
                    📰 Newspaper Advertisement  


                    Prediction is generated using a trained
                    **Decision Tree Regression Machine Learning Model**.
                    """
                )



        predict_btn.click(

            fn=predict_sales,

            inputs=[
                tv,
                radio,
                newspaper
            ],

            outputs=output
        )



# =====================================
# Launch Application
# =====================================

if __name__ == "__main__":

    app.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )
