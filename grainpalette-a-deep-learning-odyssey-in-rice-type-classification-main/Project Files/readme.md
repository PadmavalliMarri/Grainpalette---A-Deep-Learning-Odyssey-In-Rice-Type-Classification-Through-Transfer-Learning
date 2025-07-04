# 🌾 GrainPalette: Rice Type Classification

A deep learning web application that classifies rice grain images into five types using CNN with MobileNetV2 transfer learning.

## 🎯 Features

- **Deep Learning Classification**: Uses CNN with MobileNetV2 transfer learning
- **5 Rice Types**: Arborio, Basmati, Ipsala, Jasmine, Karacadag
- **Web Interface**: User-friendly Flask web application
- **Real-time Prediction**: Upload images and get instant results
- **Confidence Scores**: Detailed prediction probabilities
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### 1. Clone the Repository
\`\`\`bash
git clone <your-repo-url>
cd grainpalette-rice-classifier
\`\`\`

### 2. Set Up Environment
\`\`\`bash
python scripts/setup_environment.py
\`\`\`

## 📊 Dataset Setup

### Option 1: Kaggle Notebook (Recommended)
1. **Create a new Kaggle notebook**
2. **Add the Rice Image Dataset:**
   - Go to "Add Data" → "Datasets"
   - Search for "Rice Image Dataset" by muratkokludataset
   - Add it to your notebook
3. **Upload the `kaggle_notebook_training.py` script**
4. **Run the training script in the notebook**

### Option 2: Local Setup with Kaggle API
1. **Install Kaggle API:**
   \`\`\`bash
   pip install kaggle
   \`\`\`

2. **Set up Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json`
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

3. **Run the training script:**
   \`\`\`bash
   python train_model.py
   \`\`\`
   The script will automatically download the dataset.

### Option 3: Manual Download
1. **Visit:** https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
2. **Download and extract** to `rice_dataset/` folder
3. **Ensure structure:**
   \`\`\`
   rice_dataset/
   ├── Arborio/
   ├── Basmati/
   ├── Ipsala/
   ├── Jasmine/
   └── Karacadag/
   \`\`\`

### 4. Train the Model
\`\`\`bash
python train_model.py
\`\`\`

### 5. Run the Web Application
\`\`\`bash
python app.py
\`\`\`

Visit `http://localhost:5000` to use the application!

## 📊 Training Options

### Kaggle Notebook Training (Recommended)
- **Free GPU access** for faster training
- **Pre-installed libraries** and dependencies
- **Easy dataset access** without downloads
- **Shareable results** and visualizations

Use `kaggle_notebook_training.py` for the best Kaggle experience with:
- Dataset exploration and visualization
- Comprehensive data augmentation
- Advanced model evaluation
- Training progress visualization

### Local Training
Use `train_model.py` for local development with automatic dataset download via Kaggle API.

## 📊 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base layers + custom classification head
- **Input Size**: 224x224x3
- **Output**: 5 classes with softmax activation
- **Optimization**: Adam optimizer with learning rate scheduling

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow, Keras |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | PIL, NumPy |
| Visualization | Matplotlib |

## 📁 Project Structure

\`\`\`
grainpalette-rice-classifier/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── static/
│   ├── style.css         # CSS styles
│   └── uploads/          # Uploaded images
├── templates/
│   ├── index.html        # Home page
│   ├── result.html       # Results page
│   └── about.html        # About page
├── scripts/
│   └── setup_environment.py  # Setup script
└── rice_dataset/         # Dataset (to be downloaded)
    ├── Arborio/
    ├── Basmati/
    ├── Ipsala/
    ├── Jasmine/
    └── Karacadag/
\`\`\`

## 🎯 Use Cases

- **Farmers**: Identify seed types before cultivation
- **Agricultural Scientists**: Validate crop classification
- **Educators**: Teaching tool for agricultural studies
- **Food Industry**: Quality control and grain sorting
- **Researchers**: Rice variety analysis

## 📈 Model Performance

The model achieves high accuracy through:
- Transfer learning from ImageNet
- Data augmentation for robustness
- Fine-tuning for domain adaptation
- Comprehensive validation

## 🔧 Configuration

Key parameters in `train_model.py`:
- `IMG_SIZE`: Input image dimensions (224, 224)
- `BATCH_SIZE`: Training batch size (32)
- `EPOCHS`: Training epochs (20)
- `LEARNING_RATE`: Initial learning rate (0.0001)

## 🚀 Deployment Options

### Local Development
\`\`\`bash
python app.py
\`\`\`

### Production Deployment
- **Heroku**: Use provided `requirements.txt`
- **Docker**: Create Dockerfile for containerization
- **Cloud Platforms**: Deploy on AWS, GCP, or Azure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Rice Image Dataset from Kaggle
- TensorFlow and Keras teams
- MobileNetV2 architecture
- Flask web framework

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the setup instructions
3. Open an issue on GitHub

---

**Happy Rice Classification! 🌾**
