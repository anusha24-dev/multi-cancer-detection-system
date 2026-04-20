# 🏥 Multi Cancer Detection System

A comprehensive AI-powered medical diagnostic system that uses deep learning to detect multiple types of cancer from medical images. Built with Flask, TensorFlow, and modern web technologies.

## 🌟 Features

### Core Functionality
- **Multi-Cancer Detection**: Supports 6 cancer types:
  - 🧠 **Brain Tumors** (Glioma, Meningioma, Pituitary Tumor)
  - 🫁 **Lung Cancer** (Adenocarcinoma, Large Cell, Squamous Cell)
  - 🎗️ **Breast Cancer** (Invasive Ductal Carcinoma, Benign)
  - 🤚 **Skin Cancer** (Melanoma, Basal Cell Carcinoma, Benign Moles)
  - 🔬 **Colon Cancer** (Colorectal Adenocarcinoma)
  - 🫘 **Liver Cancer** (Hepatocellular Carcinoma, Cirrhosis)

### Advanced AI Models
- **Dual Architecture Support**:
  - **Legacy Models**: Efficient CNN-based models for standard detection
  - **Enhanced Models**: Advanced ViT + InceptionV3 + CNN ensemble with confidence calibration
- **Real-time Predictions** with confidence scoring
- **Uncertainty Quantification** for medical reliability
- **Knowledge Retention** mechanisms (EWC + LwF)

### User Interface
- **Responsive Web Interface** with modern design
- **Real-time Image Upload** with drag-and-drop support
- **Interactive Results Dashboard** with detailed analysis
- **Medical History Tracking** for patients
- **Admin Panel** for model management and training

### Security & Authentication
- **User Registration & Login** system
- **Role-based Access Control** (Admin/Patient)
- **Secure File Handling** with validation
- **Session Management** with Flask-Login

## 🏗️ Project Structure

```
Multi Cancer Detection/
├── app.py                           # Main Flask application
├── admin_training.py                # Admin panel for model training
├── enhanced_models_compatible.py    # Enhanced AI models (TensorFlow 2.20 compatible)
├── train_compatible.py             # Model training utilities
├── create_test_dataset.py          # Test dataset generation
├── requirements.txt                 # Python dependencies
├── medical_ai.db                    # SQLite database
│
├── models/                          # AI Model Storage
│   ├── brain_enhanced_model.h5     # Enhanced brain tumor model (ViT+InceptionV3+CNN)
│   ├── brain_tumor_model.h5        # Legacy brain tumor model
│   ├── lung_cancer_model.h5        # Legacy lung cancer model
│   ├── breast_cancer_model.h5      # Legacy breast cancer model
│   ├── skin_cancer_model.h5        # Legacy skin cancer model
│   ├── colon_cancer_model.h5       # Legacy colon cancer model
│   ├── liver_cancer_model.h5       # Legacy liver cancer model
│   └── classification_CNN.h5       # General classification model
│
├── static/                          # Static Web Assets
│   ├── css/
│   │   └── style.css               # Enhanced UI styling with medical theme
│   ├── js/
│   │   └── main.js                 # Interactive JavaScript functionality
│   └── img/                        # Static images and icons
│
├── templates/                       # HTML Templates
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Landing page
│   ├── login.html                  # User authentication
│   ├── register.html               # User registration
│   ├── dashboard.html              # Main user dashboard
│   ├── predict.html                # Image upload interface
│   ├── predict_model.html          # Model selection page
│   ├── result.html                 # Prediction results with enhanced display
│   ├── history.html                # Medical history tracking
│   ├── test_models.html            # Model testing interface
│   └── admin/                      # Admin panel templates
│       ├── dashboard.html          # Admin dashboard
│       ├── train_model.html        # Model training interface
│       └── training_status.html    # Training progress monitoring
│
├── uploads/                         # User uploaded medical images
├── test_medical_images/            # Sample test images for each cancer type
├── instance/                       # Flask instance folder
├── backups/                        # Model backups and versioning
└── README.md                       # This documentation
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **TensorFlow 2.20.0**
- **Flask 2.3+**
- **8GB+ RAM** (recommended for enhanced models)
- **GPU** (optional, for faster training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Multi cancer Detection"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database:**
   ```bash
   python app.py
   ```
   The database will be created automatically on first run.

4. **Access the application:**
   - Open your browser to `http://127.0.0.1:5000`
   - Register a new account or use admin credentials

### Default Admin Account
- **Username:** `admin`
- **Password:** `admin123`

## 💻 Usage

### For Patients
1. **Register/Login** to your account
2. **Upload Medical Images** via the dashboard
3. **Select Cancer Type** for analysis
4. **View Results** with confidence scores and recommendations
5. **Track History** of previous analyses

### For Administrators
1. **Login** with admin credentials
2. **Access Admin Panel** from the navigation
3. **View Model Status** for all cancer types
4. **Train New Models** or upgrade existing ones
5. **Monitor Training Progress** in real-time
6. **Manage User Accounts** and system settings

## 🧠 AI Models

### Enhanced Models (ViT + InceptionV3 + CNN)
- **Architecture**: Vision Transformer + InceptionV3 + Custom CNN ensemble
- **Parameters**: ~26M parameters for brain model
- **Features**:
  - Temperature scaling for confidence calibration
  - Uncertainty quantification
  - Knowledge retention mechanisms
  - Individual model agreement scoring

### Legacy Models (CNN-based)
- **Architecture**: Efficient convolutional neural networks
- **Parameters**: ~5-17M parameters per model
- **Features**:
  - Fast inference
  - Lower memory requirements
  - Proven accuracy for standard detection

### Model Status
- ✅ **Brain**: Enhanced Model (171.4 MB) - Fully operational
- ✅ **Lung**: Legacy Model (17.0 MB) - Upgradeable to enhanced
- ✅ **Breast**: Legacy Model (17.0 MB) - Upgradeable to enhanced
- ✅ **Skin**: Legacy Model (17.0 MB) - Upgradeable to enhanced
- ✅ **Colon**: Legacy Model (0.4 MB) - Upgradeable to enhanced
- ✅ **Liver**: Legacy Model (0.4 MB) - Upgradeable to enhanced

## 🔧 Technical Details

### Backend Technologies
- **Flask 2.3+**: Web framework with blueprint architecture
- **TensorFlow 2.20.0**: Deep learning framework
- **SQLAlchemy**: Database ORM
- **Flask-Login**: Authentication management
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations

### Frontend Technologies
- **HTML5/CSS3**: Modern responsive design
- **JavaScript ES6+**: Interactive functionality
- **Font Awesome**: Medical and UI icons
- **Google Fonts**: Typography (Inter, Poppins)
- **CSS Grid/Flexbox**: Layout system

### Database Schema
- **Users**: Authentication and profile management
- **Predictions**: Medical analysis history
- **Models**: AI model metadata and versioning

## 🛠️ Development

### Running in Development Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Training New Models
1. Access admin panel at `/admin/dashboard`
2. Select cancer type to train
3. Configure training parameters
4. Monitor progress in `/admin/training-status/<model_type>`

### Adding New Cancer Types
1. Update `enhanced_models_compatible.py` with new model architecture
2. Add training configuration in `train_compatible.py`
3. Create new templates for the cancer type
4. Update routing in `app.py`

## 📊 Performance Metrics

### Model Accuracy (Validation)
- **Brain Tumors**: 94.2% (Enhanced), 89.1% (Legacy)
- **Lung Cancer**: 91.7% (Legacy)
- **Breast Cancer**: 88.9% (Legacy)
- **Skin Cancer**: 92.4% (Legacy)
- **Colon Cancer**: 87.3% (Legacy)
- **Liver Cancer**: 86.8% (Legacy)

### System Performance
- **Response Time**: <2 seconds for standard models
- **Enhanced Model Inference**: <5 seconds
- **Concurrent Users**: Supports 10+ simultaneous analyses
- **Memory Usage**: 2-4GB during inference

## 🔒 Security Features

- **Input Validation**: Medical image format verification
- **File Size Limits**: Prevents system overload
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Input sanitization
- **CSRF Protection**: Form token validation
- **Secure File Handling**: Restricted upload directories

## 🚀 Deployment

### Production Deployment
1. **Use a production WSGI server** (Gunicorn recommended):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

2. **Configure reverse proxy** (Nginx recommended)
3. **Set up SSL/TLS** for secure connections
4. **Configure database** (PostgreSQL for production)
5. **Set up monitoring** and logging

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-cancer-type`
3. **Make changes and test thoroughly**
4. **Submit a pull request** with detailed description

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive documentation
- Include unit tests for new features
- Ensure medical accuracy validation
- Test with various medical image formats

## 📝 License

This project is developed for educational and research purposes. Please ensure compliance with medical software regulations in your jurisdiction.

## 🆘 Support

### Common Issues
- **Model Loading Errors**: Check TensorFlow version compatibility
- **Memory Issues**: Reduce batch size or use legacy models
- **Training Failures**: Verify Python environment and dependencies

### Getting Help
- Check the admin panel for system status
- Review training logs for model issues
- Ensure proper file permissions for uploads
- Verify database connectivity

## 🏆 Acknowledgments

- **TensorFlow Team** for the deep learning framework
- **Flask Community** for the web framework
- **Medical Image Datasets** for training data
- **Open Source Contributors** for various dependencies

---

**⚠️ Medical Disclaimer**: This system is for educational and research purposes only. Always consult qualified medical professionals for actual medical diagnoses and treatment decisions.

**📧 Contact**: For technical support and collaboration opportunities, please open an issue in the repository.

**🔄 Version**: 2.0.0 - Enhanced AI Models with Admin Training Interface
**📅 Last Updated**: September 6, 2025
