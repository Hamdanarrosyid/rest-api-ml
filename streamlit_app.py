import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import random
import json

# Configuration
FLASK_API_URL = "http://localhost:5000/predict" # Your Flask API endpoint

def get_random_color_tuple():
    """Generates a random RGB color tuple."""
    r = random.randint(25, 224) # Avoid too dark/light
    g = random.randint(25, 224)
    b = random.randint(25, 224)
    return (r, g, b)

def draw_segmentations_on_image(image, segmentations_data):
    """
    Draws segmentation polygons and labels on the image.

    Args:
        image (PIL.Image): The original image.
        segmentations_data (list): A list of segmentation dicts from the API.

    Returns:
        PIL.Image: A new image with segmentations drawn.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_with_masks = image.copy()
    draw = ImageDraw.Draw(img_with_masks, "RGBA") # Use RGBA for transparent fill

    for seg in segmentations_data:
        points = seg.get("polygon_points_xy")
        if not points or len(points) < 2:
            continue

        class_name = seg.get("class_name", "N/A")
        confidence = seg.get("confidence", 0.0)

        # Convert list of [x,y] lists to list of (x,y) tuples for PIL
        polygon_tuples = [tuple(p) for p in points]

        base_color = get_random_color_tuple()
        outline_color = base_color
        fill_color = base_color + (int(0.35 * 255),) # Add alpha for fill (approx 35% opacity)

        line_width = max(1, min(3, round(image.width / 300)))
        draw.polygon(polygon_tuples, outline=outline_color, fill=fill_color, width=line_width)

        # Draw label
        label = f"{class_name} ({confidence*100:.0f}%)"
        font_size = max(10, min(14, round(image.width / 50)))
        # Note: PIL's default font is basic. For better fonts, you'd load a .ttf file.
        # For simplicity, we use the default font.
        # text_position = (points[0][0] + 5, points[0][1] - 5 - font_size) # Adjust for text height
        
        # A simple way to position text near the first point of the polygon
        text_x = points[0][0] + 5
        text_y = points[0][1] - 5 - font_size

        # Basic boundary checks for text
        if text_y < 0: text_y = 5
        if text_x < 0: text_x = 5
        # More sophisticated text placement would require text_bbox if using specific fonts

        draw.text((text_x, text_y), label, fill=outline_color) # Using outline_color for text

    return img_with_masks

st.set_page_config(layout="wide", page_title="Image Segmentation Inference")
st.title("🖼️ Image Segmentation with YOLO API")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("🔍 Run Inference", use_container_width=True):
        with st.spinner("🧠 Processing image and running inference..."):
            try:
                files = {'image': (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(FLASK_API_URL, files=files, timeout=60) # Added timeout
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

                api_result = response.json()

                with col2:
                    st.subheader("Segmentation Result")
                    if api_result.get("segmentations") and len(api_result["segmentations"]) > 0:
                        segmented_image = draw_segmentations_on_image(original_image, api_result["segmentations"])
                        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
                    elif api_result.get("error"):
                        st.error(f"API Error: {api_result['error']} - Details: {api_result.get('details', 'N/A')}")
                        st.image(original_image, caption="Original Image (No Segmentation)", use_column_width=True) # Show original if error
                    else:
                        st.info("No segmentations found in the image by the model.")
                        st.image(original_image, caption="Original Image (No Segmentation)", use_column_width=True) # Show original if no segments

                st.subheader("API JSON Output")
                # Create a copy for display, removing polygon_points_xy
                display_json = json.loads(json.dumps(api_result)) # Deep copy
                if "segmentations" in display_json and isinstance(display_json["segmentations"], list):
                    for seg in display_json["segmentations"]:
                        if "polygon_points_xy" in seg:
                            del seg["polygon_points_xy"]
                st.json(display_json)

            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the API at {FLASK_API_URL}. Please ensure the Flask API server is running and accessible.")
            except requests.exceptions.Timeout:
                st.error(f"Request Timeout: The request to the API at {FLASK_API_URL} timed out. The server might be too busy or the image too large for the current timeout setting (60s).")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.reason}")
                try:
                    error_details = e.response.json()
                    st.json({"error_details_from_api": error_details})
                except json.JSONDecodeError:
                    st.text("Could not parse error details from API response.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    else:
        with col2:
            st.info("Click 'Run Inference' in the sidebar to see the results.")
else:
    st.info("☝️ Upload an image using the sidebar to get started.")

st.sidebar.markdown("---")
st.sidebar.markdown("Ensure your Flask API server is running at:")
st.sidebar.code(FLASK_API_URL, language=None)