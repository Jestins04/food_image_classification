from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import os
import pandas as pd
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import subprocess
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter

app = Flask(__name__)
app.secret_key = "recipe_secret"

# ================== DATABASE ==================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="recipe_finder",
    charset="utf8"
)
cursor = db.cursor(dictionary=True)

# ================== CONFIG ==================
UPLOAD_FOLDER = "static/uploads"
IMAGE_SIZE = (224, 224)  # Swin Transformer input size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# ================== HOME ==================
@app.route("/")
def index():
    return render_template("index.html")

# ======================================================
# ADMIN SECTION
# ======================================================
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        if u == ADMIN_USER and p == ADMIN_PASS:
            session["admin"] = True
            return redirect("/admin/dashboard")
        else:
            flash("Invalid admin login")
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin/login")
    return render_template("admin_dashboard.html")

# Paths
INPUT_DIR = "static/images"
PREPROCESS_DIR = "static/preprocessed"
BINARY_DIR = "static/binary"
SEGMENT_DIR = "static/segmented"
FEATURE_DIR = "static/features"

for folder in [PREPROCESS_DIR, BINARY_DIR, SEGMENT_DIR, FEATURE_DIR]:
    os.makedirs(folder, exist_ok=True)

IMG_SIZE = (224, 224)
IMAGES_PER_CLASS = 1
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def resize_image(image):
    """Resize image to IMG_SIZE"""
    return cv2.resize(image, IMG_SIZE)


@app.route('/train')
def train():
    dataset = []
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if os.path.isdir(class_path):
            imgs = [os.path.join('static/images', class_name, f)
                    for f in os.listdir(class_path)
                    if f.lower().endswith(valid_extensions)]
            dataset.extend(imgs[:IMAGES_PER_CLASS])
    return render_template('dataset.html', images=dataset)


# ---------------- PROCESS 1 ----------------
@app.route('/process1')
def process1_grayscale():
    processed_images = []
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_extensions)]
        images = images[:IMAGES_PER_CLASS]

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Resize first
            image = resize_image(image)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            out_name = f"{class_name}_{idx}.jpg"
            out_path = os.path.join(PREPROCESS_DIR, out_name)
            cv2.imwrite(out_path, gray)
            processed_images.append(out_path.replace("static/", "static/"))
    return render_template('process1.html', images=processed_images)

# ---------------- PROCESS 2 ----------------
@app.route('/process2')
def process2_binary():
    processed_images = []
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_extensions)]
        images = images[:IMAGES_PER_CLASS]

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Resize first
            image = resize_image(image)

            # Grayscale then binary
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            out_name = f"{class_name}_{idx}.jpg"
            out_path = os.path.join(BINARY_DIR, out_name)
            cv2.imwrite(out_path, binary)
            processed_images.append(out_path.replace("static/", "static/"))
    return render_template('process2.html', images=processed_images)

# ---------------- PROCESS 3 ----------------
@app.route('/process3')
def process3_segmentation():
    processed_images = []
    K = 2
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_extensions)]
        images = images[:IMAGES_PER_CLASS]

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Resize first
            image = resize_image(image)

            # K-means segmentation
            pixels = image.reshape((-1,3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.astype(np.uint8)
            segmented = centers[labels.flatten()].reshape(image.shape)

            out_name = f"{class_name}_{idx}.jpg"
            out_path = os.path.join(SEGMENT_DIR, out_name)
            cv2.imwrite(out_path, segmented)
            processed_images.append(out_path.replace("static/", "static/"))
    return render_template('process3.html', images=processed_images)

# ---------------- PROCESS 4 ----------------
@app.route('/process4')
def process4_features():
    processed_images = []
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_extensions)]
        images = images[:IMAGES_PER_CLASS]

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Resize first
            image = resize_image(image)

            # Edge detection overlay
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            overlay = image.copy()
            overlay[edges != 0] = [0, 255, 0]

            out_name = f"{class_name}_{idx}.jpg"
            out_path = os.path.join(FEATURE_DIR, out_name)
            cv2.imwrite(out_path, overlay)
            processed_images.append(out_path.replace("static/", "static/"))
    return render_template('process4.html', images=processed_images)

@app.route("/classify")
def classify():
    if "admin" not in session:
        return redirect("/admin/login")

    # ===== Load your training logs =====
    # For demo, we'll simulate it. 
    # If you train offline, you should save loss/accuracy arrays to pickle or CSV after training.
    try:
        import pickle
        with open("train_logs.pkl", "rb") as f:
            logs = pickle.load(f)
            losses = logs['losses']
            accuracies = logs['accuracies']
    except:
        # Demo data if logs not found
        losses = [2.3, 2.0, 1.7, 1.5, 1.2]
        accuracies = [0.1, 0.3, 0.5, 0.6, 0.7]

    # ===== Plot loss vs epochs =====
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(1,len(losses)+1), losses, marker='o', color='red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Epochs")
    ax.grid(True)

    # Save figure to PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    loss_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # ===== Plot accuracy vs epochs =====
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(range(1,len(accuracies)+1), accuracies, marker='o', color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy vs Epochs")
    ax2.grid(True)

    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    acc_graph = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    images_base = "static/images"

    class_labels = []
    class_values = []

    # Loop over each folder (class) and count images
    for folder_name in os.listdir(images_base):
        folder_path = os.path.join(images_base, folder_name)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            class_labels.append(folder_name)
            class_values.append(num_images)  # already Python int

    return render_template("classify.html",
                           loss_graph=loss_graph,
                           acc_graph=acc_graph,
                           class_labels=class_labels,
                           class_values=class_values)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/")

# ======================================================
# USER SECTION
# ======================================================
@app.route("/user/register", methods=["GET","POST"])
def user_register():
    if request.method == "POST":
        data = (
            request.form["name"],
            request.form["email"],
            request.form["mobile"],
            request.form["username"],
            request.form["password"]
        )
        cursor.execute(
            "INSERT INTO users(name,email,mobile,username,password) VALUES(%s,%s,%s,%s,%s)",
            data
        )
        db.commit()
        flash("Registered successfully!")
        return redirect("/user/login")
    return render_template("user_register.html")

@app.route("/user/login", methods=["GET","POST"])
def user_login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (u,p)
        )
        user = cursor.fetchone()
        if user:
            session["user"] = user["username"]
            return redirect("/user/dashboard")
        else:
            flash("Invalid login")
    return render_template("user_login.html")

@app.route("/user/dashboard")
def user_dashboard():
    if "user" not in session:
        return redirect("/user/login")
    return render_template("user_dashboard.html")

def generate_food_recommendation(recipe):
    name = recipe.get("recipe_name", "")
    calories = recipe.get("calories_per_serving", 0)
    veg = recipe.get("veg_nonveg", "")
    spicy = recipe.get("spicy_level", "")
    diabetic = recipe.get("diabetic_friendly", "")
    heart = recipe.get("heart_friendly", "")

    info = f"{name} is a "

    # Veg / Non-veg
    if veg == "veg":
        info += "vegetarian dish "
    else:
        info += "non-vegetarian dish "

    # Calories
    if calories < 250:
        info += "that is low in calories, making it a good option for weight management. "
    elif calories < 400:
        info += "with moderate calories, suitable for a balanced diet. "
    else:
        info += "that is relatively high in calories, so it should be consumed in moderation. "

    # Spicy
    if spicy == "hot":
        info += "It is quite spicy and may boost metabolism but may not suit sensitive stomachs. "
    elif spicy == "medium":
        info += "It has a moderate spice level suitable for most people. "
    else:
        info += "It is mildly spiced and easy to digest. "

    # Health
    if diabetic == "yes":
        info += "This dish is diabetic-friendly and helps maintain stable blood sugar. "
    else:
        info += "Not ideal for diabetics if consumed frequently. "

    if heart == "yes":
        info += "It is heart-friendly and supports cardiovascular health. "
    else:
        info += "Frequent consumption may not support heart health. "

    # Allergens
    allergens = []

    if recipe.get("contains_gluten") == "yes":
        allergens.append("gluten")
    if recipe.get("contains_dairy") == "yes":
        allergens.append("dairy")
    if recipe.get("contains_eggs") == "yes":
        allergens.append("eggs")
    if recipe.get("contains_nuts") == "yes":
        allergens.append("nuts")
    if recipe.get("contains_seafood") == "yes":
        allergens.append("seafood")

    if allergens:
        info += "Contains " + ", ".join(allergens) + ". "

    return info

# ================== IMAGE PREDICTION ==================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/user/login")

    file = request.files["image"]
    filename = secure_filename(file.filename)
    
    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Load model and label encoder
    import pandas as pd
    import pickle
    from PIL import Image
    import timm
    import torch
    from torchvision import transforms

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("dataset.csv")
    le = pickle.load(open("label_encoder.pkl", "rb"))
    NUM_CLASSES = df["image_class_label"].nunique()

    # Transform for Swin
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load model
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("model_swin.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Prepare image
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = le.inverse_transform([pred.item()])[0]

    recipe = df[df["image_class_label"] == label].iloc[0].to_dict()

    # 🔥 ADD HEALTH RECOMMENDATION
    food_info = generate_food_recommendation(recipe)

    return render_template("result.html",
                           r=recipe,
                           img=filename,
                           food_info=food_info)

@app.route("/predict_text", methods=["POST"])
def predict_text():
    if "user" not in session:
        return redirect("/user/login")

    import pandas as pd

    user_input = request.form.get("ingredients")

    if not user_input:
        flash("Please enter ingredients")
        return redirect("/user/dashboard")

    df = pd.read_csv("dataset.csv")

    user_ingredients = [i.strip().lower() for i in user_input.split(",")]

    results = []

    for _, row in df.iterrows():
        recipe_ingredients = str(row.get("ingredients", "")).lower()

        match_count = sum(1 for ing in user_ingredients if ing in recipe_ingredients)

        if match_count > 0:
            results.append((match_count, row))

    results = sorted(results, key=lambda x: x[0], reverse=True)

    if not results:
        flash("No recipes found")
        return redirect("/user/dashboard")

    # MULTIPLE RESULTS
    top_recipes = [r[1].to_dict() for r in results[:5]]

    # 🔥 ADD HEALTH INFO FOR EACH RECIPE
    recipes_with_info = []
    for recipe in top_recipes:
        recipe["food_info"] = generate_food_recommendation(recipe)
        recipes_with_info.append(recipe)

    return render_template("result.html",
                           recipes=recipes_with_info,
                           r=None,
                           img=None)


@app.route("/user/logout")
def user_logout():
    session.pop("user", None)
    return redirect("/")

@app.route("/logout")
def logout():
    return redirect("/")
# ==================
if __name__ == "__main__":
    app.run(debug=True)
