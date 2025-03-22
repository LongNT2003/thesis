import streamlit as st
import streamlit_shortcuts
import os
import pandas as pd
import argparse


# Argument parsing for command-line execution
def get_arguments():
    parser = argparse.ArgumentParser(description="Streamlit labeling tool")
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the main folder containing product folders",
    )
    args = parser.parse_args()
    return args.folder


# Get MAIN_FOLDER from command line
MAIN_FOLDER = get_arguments()
folder_name = os.path.basename(MAIN_FOLDER)
RESULT_FILE = f"review_labels_{folder_name}.csv"

# Streamlit UI
st.sidebar.title("Product Labeling Tool")
st.sidebar.write(f"Labeling for folder: `{MAIN_FOLDER}`")
st.sidebar.write(f"Results will be saved in: `{RESULT_FILE}`")

# Your Streamlit app logic goes here


# Load existing labels if available
if os.path.exists(RESULT_FILE):
    df_labels = pd.read_csv(RESULT_FILE)
else:
    df_labels = pd.DataFrame(columns=["product", "review", "label_correct"])


# Get all product folders
def get_products():
    all_products = [
        f
        for f in os.listdir(MAIN_FOLDER)
        if os.path.isdir(os.path.join(MAIN_FOLDER, f))
    ]
    labeled_products = (
        set(df_labels["product"].unique()) if not df_labels.empty else set()
    )
    return [p for p in all_products if p not in labeled_products]


# Load product image & review images
def load_images(product_folder):
    product_path = os.path.join(MAIN_FOLDER, product_folder)
    images = [
        img
        for img in os.listdir(product_path)
        if img.lower().endswith(("png", "jpg", "jpeg"))
    ]
    product_image = images[0] if images else None

    review_folder = [
        f
        for f in os.listdir(product_path)
        if os.path.isdir(os.path.join(product_path, f))
    ][0]
    review_folder = os.path.join(product_path, review_folder)
    review_images = []
    if os.path.exists(review_folder):
        review_images = [
            os.path.join(review_folder, img)
            for img in os.listdir(review_folder)
            if img.lower().endswith(("png", "jpg", "jpeg"))
        ][:1]
    return product_image, review_images


if "product" not in st.session_state:
    st.session_state.product = get_products()
st.session_state.current_index = st.session_state.get("current_index", 0)

if st.session_state.current_index >= len(st.session_state.product):
    st.write("### Labeling Complete!")
    st.stop()

current_product = st.session_state.product[st.session_state.current_index]
product_image, review_images = load_images(current_product)
st.write(f"## Product: {current_product}")

left, right = st.columns(2, vertical_alignment="top")
if product_image:
    left.image(
        os.path.join(MAIN_FOLDER, current_product, product_image),
        caption="Product Image",
        width=300,
    )

for review_img in review_images:
    right.image(review_img, caption=f"Review Image: {review_img}", width=300)
    correct = streamlit_shortcuts.button(
        "✅ Correct",
        key="correct",
        shortcut=".",
        on_click=lambda: st.success("Button clicked!"),
    )
    incorrect = streamlit_shortcuts.button(
        "❌ Incorrect",
        key="incorrect",
        shortcut=",",
        on_click=lambda: st.success("Button clicked!"),
    )

    if correct or incorrect:
        label = "Correct" if correct else "Incorrect"
        new_entry = pd.DataFrame(
            [[current_product, review_img, label]], columns=df_labels.columns
        )
        df_labels = pd.concat([df_labels, new_entry], ignore_index=True)
        df_labels.to_csv(RESULT_FILE, index=False)

        if (
            review_img == review_images[-1]
        ):  # Move to next product if last review image is labeled
            st.session_state.current_index += 1
            st.rerun()

# BACK BUTTON: Remove last entry and go back
if st.session_state.current_index > 0:
    if streamlit_shortcuts.button(
        "⬅️ Back",
        key="back",
        shortcut="m",
        on_click=lambda: st.success("Button clicked!"),
    ):
        if not df_labels.empty:
            df_labels = df_labels.iloc[:-1]  # Remove last row
            df_labels.to_csv(RESULT_FILE, index=False)  # Save updated CSV

        st.session_state.current_index -= 1  # Move back
        st.rerun()
