import os
import gradio as gr
import joblib
import pandas as pd


# ==========================
# Load Model
# ==========================

model = joblib.load("Sales_Prediction_Model.pkl")


# Get original training columns
try:
    FEATURES = list(model.feature_names_in_)
except:
    FEATURES = ["TV", "Radio", "Newspaper"]


print("Model expects:", FEATURES)



# ==========================
# Prediction Function
# ==========================

def predict_sales(tv, radio, newspaper):

    try:

        values = {
            "TV": tv,
            "Radio": radio,
            "Newspaper": newspaper
        }


        # Arrange columns exactly like training
        input_data = pd.DataFrame(
            [[values[col] for col in FEATURES]],
            columns=FEATURES
        )


        prediction = model.predict(input_data)[0]


        return f"📈 Predicted Sales: {prediction:.2f} units"


    except Exception as e:

        return f"Error: {str(e)}"



# ==========================
# Gradio Interface
# ==========================

with gr.Blocks(title="Sales Prediction") as app:


    gr.Markdown(
        """
        # 📊 Sales Prediction using Decision Tree

        Enter advertisement budget values.
        """
    )


    with gr.Row():

        with gr.Column():

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


            button = gr.Button(
                "Predict Sales"
            )


        with gr.Column():

            output = gr.Textbox(
                label="Prediction Result"
            )


    button.click(
        predict_sales,
        inputs=[
            tv,
            radio,
            newspaper
        ],
        outputs=output
    )



# ==========================
# Run App
# ==========================

if __name__ == "__main__":

    app.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
