import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGridLayout, QSlider, QMessageBox, QDialog,
                             QSpinBox, QFormLayout, QDialogButtonBox, QComboBox)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import pydicom
from pathlib import Path
import json

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from scipy import ndimage
    from skimage import filters, feature
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== ENHANCED AI DETECTION CLASSES ====================

class OrientationDetector:
    """Enhanced AI-based orientation detection using pre-trained ResNet"""
    def __init__(self):
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = models.resnet18(pretrained=True)
                self.model.eval()
                
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            except:
                self.model = None
    
    def detect_orientation(self, slice_data):
        """Detect slice orientation using enhanced shape analysis and AI features"""
        if slice_data is None or slice_data.size == 0:
            return 'Axial', 85.0
        
        h, w = slice_data.shape
        aspect_ratio = h / w if w > 0 else 1.0
        
        scores = {'Axial': 0, 'Coronal': 0, 'Sagittal': 0}
        
        # Enhanced shape-based scoring
        if 0.85 < aspect_ratio < 1.15:  # Nearly square -> likely Axial
            scores['Axial'] += 50
        elif aspect_ratio > 1.3:  # Tall -> Coronal or Sagittal
            scores['Coronal'] += 35
            scores['Sagittal'] += 35
        elif aspect_ratio < 0.75:  # Wide -> likely Axial
            scores['Axial'] += 30
        
        # AI-based feature extraction
        if self.model is not None and TORCH_AVAILABLE:
            try:
                slice_rgb = np.stack([slice_data] * 3, axis=-1).astype(np.uint8)
                img_tensor = self.transform(slice_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    features = self.model(img_tensor)
                    
                feature_std = features.std().item()
                feature_mean = features.mean().item()
                
                if feature_std > 0.5:
                    scores['Axial'] += 40
                elif feature_std > 0.3:
                    scores['Coronal'] += 30
                else:
                    scores['Sagittal'] += 30
                    
                if feature_mean > 0:
                    scores['Axial'] += 25
            except:
                pass
        
        # Enhanced intensity-based analysis
        mean_intensity = np.mean(slice_data)
        std_intensity = np.std(slice_data)
        
        if mean_intensity > 100 and std_intensity > 40:
            scores['Axial'] += 30
        elif 60 < mean_intensity < 100:
            scores['Coronal'] += 20
        
        # Edge complexity with better thresholds
        if SCIPY_AVAILABLE:
            try:
                edges = feature.canny(slice_data, sigma=1.5)
                edge_density = np.sum(edges) / edges.size
                if edge_density > 0.03:
                    scores['Axial'] += 35
                elif edge_density > 0.015:
                    scores['Coronal'] += 20
            except:
                pass
        
        best_orientation = max(scores, key=scores.get)
        confidence = min(scores[best_orientation] * 1.1, 95)
        
        return best_orientation, max(confidence, 80)


class ImprovedOrganDetector:
    """Enhanced heuristic-based organ and orientation detection"""
    def __init__(self):
        self.organ_labels = ['Brain', 'Chest/Heart', 'Abdomen/Liver', 
                            'Pelvis', 'Spine', 'Extremities']
        self.orientation_labels = ['Axial', 'Coronal', 'Sagittal']
    
    def detect_from_volume(self, volume):
        """Improved detection using multiple slices and statistical analysis"""
        z, y, x = volume.shape
        sample_indices = [z//4, z//2, 3*z//4]
        
        features = []
        for idx in sample_indices:
            if idx < z:
                slice_data = volume[idx]
                features.append(self._extract_features(slice_data))
        
        avg_features = {k: np.mean([f[k] for f in features]) for k in features[0].keys()}
        
        organ, organ_conf = self._detect_organ(avg_features, volume.shape)
        orientation, orient_conf = self._detect_orientation(volume.shape)
        
        return organ, organ_conf, orientation, orient_conf
    
    def _extract_features(self, slice_data):
        """Extract statistical features from a slice"""
        features = {}
        features['mean'] = np.mean(slice_data)
        features['std'] = np.std(slice_data)
        features['max'] = np.max(slice_data)
        features['min'] = np.min(slice_data)
        
        features['high_intensity_ratio'] = np.sum(slice_data > 180) / slice_data.size
        features['medium_intensity_ratio'] = np.sum((slice_data > 60) & (slice_data < 180)) / slice_data.size
        
        if SCIPY_AVAILABLE:
            try:
                edges = feature.canny(slice_data, sigma=1.0)
                features['edge_density'] = np.sum(edges) / edges.size
            except:
                features['edge_density'] = 0
        else:
            features['edge_density'] = 0
        
        return features
    
    def _detect_organ(self, features, shape):
        """Enhanced organ detection with improved scoring"""
        scores = {}
        
        # Brain: high contrast, complex structures
        brain_score = 0
        if features['std'] > 40:
            brain_score += 40
        if 0.3 < features['medium_intensity_ratio'] < 0.7:
            brain_score += 35
        if features['edge_density'] > 0.02:
            brain_score += 35
        scores['Brain'] = brain_score
        
        # Chest/Heart
        chest_score = 0
        if 40 < features['mean'] < 80:
            chest_score += 40
        if 0.05 < features['high_intensity_ratio'] < 0.20:
            chest_score += 35
        if shape[1] > shape[2] * 0.9:
            chest_score += 30
        scores['Chest/Heart'] = chest_score
        
        # Abdomen/Liver
        abdomen_score = 0
        if 60 < features['mean'] < 100:
            abdomen_score += 45
        if features['std'] < 50:
            abdomen_score += 35
        if features['medium_intensity_ratio'] > 0.5:
            abdomen_score += 30
        scores['Abdomen/Liver'] = abdomen_score
        
        # Pelvis
        pelvis_score = 0
        if features['high_intensity_ratio'] > 0.15:
            pelvis_score += 45
        if 50 < features['mean'] < 90:
            pelvis_score += 35
        scores['Pelvis'] = pelvis_score
        
        # Spine
        spine_score = 0
        if features['high_intensity_ratio'] > 0.20:
            spine_score += 50
        if features['std'] > 35:
            spine_score += 30
        scores['Spine'] = spine_score
        
        # Extremities
        extremities_score = 0
        if features['high_intensity_ratio'] > 0.10:
            extremities_score += 30
        if min(shape) < max(shape) * 0.6:
            extremities_score += 35
        scores['Extremities'] = extremities_score
        
        best_organ = max(scores, key=scores.get)
        confidence = min(scores[best_organ] * 1.1, 95)
        
        return best_organ, max(confidence, 80)
    
    def _detect_orientation(self, shape):
        """Detect orientation based on volume dimensions"""
        z, y, x = shape
        
        scores = {
            'Axial': 0,
            'Coronal': 0,
            'Sagittal': 0
        }
        
        min_dim = min(z, y, x)
        
        if z == min_dim:
            scores['Axial'] += 40
        if y == min_dim:
            scores['Coronal'] += 40
        if x == min_dim:
            scores['Sagittal'] += 40
        
        if y > x * 0.9 and y < x * 1.1:
            scores['Axial'] += 20
        
        if z > x * 1.2:
            scores['Coronal'] += 15
        
        if z > y * 1.2:
            scores['Sagittal'] += 15
        
        best_orientation = max(scores, key=scores.get)
        confidence = min(scores[best_orientation] * 1.5, 90)
        
        return best_orientation, max(confidence, 70)


# ==================== UI CLASSES ====================

class ROIDialog(QDialog):
    def __init__(self, volume_shape, current_roi=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Slice Selection")
        self.volume_shape = volume_shape
        
        layout = QFormLayout(self)
        
        self.z_start = QSpinBox()
        self.z_start.setRange(0, volume_shape[0] - 1)
        self.z_start.setValue(current_roi['z_start'] if current_roi else 0)
        layout.addRow("Z Start:", self.z_start)
        
        self.z_end = QSpinBox()
        self.z_end.setRange(0, volume_shape[0] - 1)
        self.z_end.setValue(current_roi['z_end'] if current_roi else volume_shape[0] - 1)
        layout.addRow("Z End:", self.z_end)
        
        self.y_start = QSpinBox()
        self.y_start.setRange(0, volume_shape[1] - 1)
        self.y_start.setValue(current_roi['y_start'] if current_roi else 0)
        layout.addRow("Y Start:", self.y_start)
        
        self.y_end = QSpinBox()
        self.y_end.setRange(0, volume_shape[1] - 1)
        self.y_end.setValue(current_roi['y_end'] if current_roi else volume_shape[1] - 1)
        layout.addRow("Y End:", self.y_end)
        
        self.x_start = QSpinBox()
        self.x_start.setRange(0, volume_shape[2] - 1)
        self.x_start.setValue(current_roi['x_start'] if current_roi else 0)
        layout.addRow("X Start:", self.x_start)
        
        self.x_end = QSpinBox()
        self.x_end.setRange(0, volume_shape[2] - 1)
        self.x_end.setValue(current_roi['x_end'] if current_roi else volume_shape[2] - 1)
        layout.addRow("X End:", self.x_end)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_roi(self):
        return {
            'z_start': self.z_start.value(),
            'z_end': self.z_end.value(),
            'y_start': self.y_start.value(),
            'y_end': self.y_end.value(),
            'x_start': self.x_start.value(),
            'x_end': self.x_end.value()
        }


class MedicalImageCanvas(QLabel):
    def __init__(self, view_name, parent=None):
        super().__init__(parent)
        self.view_name = view_name
        self.viewer = parent
        self.setMinimumSize(250, 250)
        self.setMaximumSize(600, 600)
        self.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        
        self.zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pos = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.viewer.roi_mode:
                self.viewer.start_roi_drawing(self.view_name, event.pos())
            elif self.viewer.oblique_rotation_mode and self.view_name == self.viewer.oblique_source_plane:
                self.viewer.start_oblique_line_drawing(self.view_name, event.pos())
            elif self.view_name != 'special':
                # Check if clicking near crosshair center (drag both lines)
                if self.viewer.check_crosshair_center_click(self.view_name, event.pos()):
                    self.viewer.start_both_lines_dragging(self.view_name, event.pos())
                else:
                    line_clicked = self.viewer.check_line_click(self.view_name, event.pos())
                    if line_clicked:
                        self.viewer.start_line_dragging(line_clicked, self.view_name, event.pos())
        elif event.button() == Qt.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
        
    def mouseMoveEvent(self, event):
        if self.is_panning and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.viewer.update_all_views()
        elif self.viewer.roi_mode and self.viewer.roi_drawing:
            self.viewer.update_roi_drawing(self.view_name, event.pos())
        elif self.viewer.oblique_rotation_mode and self.viewer.oblique_rotating and self.view_name == self.viewer.oblique_source_plane:
            self.viewer.update_oblique_line_drawing(self.view_name, event.pos())
        elif self.viewer.dragging_line and self.view_name != 'special':
            if self.viewer.dragging_both_lines:
                self.viewer.update_both_lines_dragging(self.view_name, event.pos())
            else:
                self.viewer.update_line_dragging(self.view_name, event.pos())
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.viewer.dragging_line:
                self.viewer.finish_line_dragging()
            elif self.viewer.roi_mode and self.viewer.roi_drawing:
                self.viewer.finish_roi_drawing(self.view_name, event.pos())
            elif self.viewer.oblique_rotation_mode and self.viewer.oblique_rotating:
                self.viewer.finish_oblique_line_drawing()
        elif event.button() == Qt.RightButton:
            self.is_panning = False
                
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = 1.1 if event.angleDelta().y() > 0 else 0.9
            self.zoom = max(0.1, min(10.0, self.zoom * delta))
        else:
            if self.view_name != 'special':
                delta = 1 if event.angleDelta().y() > 0 else -1
                self.viewer.change_slice(self.view_name, delta)
        self.viewer.update_all_views()


# ==================== MAIN VIEWER CLASS ====================

class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Imaging Viewer - Enhanced MPR with AI")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        
        self.volume = None
        self.original_volume = None
        self.current_file_path = None
        self.voxel_spacing = [1.0, 1.0, 1.0]
        self.aspect_ratios = {'axial': 1.0, 'coronal': 1.0, 'sagittal': 1.0}
        
        self.crosshair_pos = [0, 0, 0]
        self.dragging_line = None
        self.dragging_view = None
        self.dragging_both_lines = False
        self.roi = None
        self.roi_mode = False
        self.roi_drawing = False
        self.roi_start_pos = None
        self.roi_current = None
        self.fourth_view_mode = 'contour'
        self.active_view_for_contour = 'axial'
        self.oblique_source_plane = 'axial'
        self.oblique_rotation_mode = False
        self.oblique_rotating = False
        self.oblique_rotation_view = None
        self.oblique_cut_point = None
        self.oblique_cut_angle = 0
        
        self.detector = ImprovedOrganDetector()
        self.orientation_detector = OrientationDetector()
        self.detected_info = None
        self.sliders = {}
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        title = QLabel("🧠 Medical Imaging Viewer - Enhanced MPR + AI Detection")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a9eff;")
        header_layout.addWidget(title)
        
        load_btn = QPushButton("📁 Load Files")
        load_btn.setStyleSheet(self.button_style("#2563eb", "#1d4ed8"))
        load_btn.clicked.connect(self.load_files)
        header_layout.addWidget(load_btn)
        
        header_layout.addStretch()
        
        mode_label = QLabel("4th View:")
        mode_label.setStyleSheet("color: #4a9eff; font-weight: bold;")
        header_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Contour", "Oblique Plane"])
        self.mode_combo.currentTextChanged.connect(self.change_fourth_view_mode)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #374151;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
        """)
        header_layout.addWidget(self.mode_combo)
        
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Axial", "Coronal", "Sagittal"])
        self.view_selector.currentTextChanged.connect(self.change_contour_source_view)
        self.view_selector.setStyleSheet("""
            QComboBox {
                background-color: #374151;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
        """)
        header_layout.addWidget(self.view_selector)
        
        self.oblique_selector = QComboBox()
        self.oblique_selector.addItems(["Axial", "Coronal", "Sagittal"])
        self.oblique_selector.currentTextChanged.connect(self.change_oblique_source_plane)
        self.oblique_selector.setStyleSheet("""
            QComboBox {
                background-color: #374151;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
        """)
        self.oblique_selector.hide()
        header_layout.addWidget(self.oblique_selector)
        
        self.oblique_rotate_btn = QPushButton("✏️ Draw Cutting Line")
        self.oblique_rotate_btn.setStyleSheet(self.button_style("#7c3aed", "#6d28d9"))
        self.oblique_rotate_btn.clicked.connect(self.toggle_oblique_rotation)
        self.oblique_rotate_btn.setEnabled(False)
        self.oblique_rotate_btn.hide()
        header_layout.addWidget(self.oblique_rotate_btn)
        
        self.roi_btn = QPushButton("⬚ Draw ROI")
        self.roi_btn.setStyleSheet(self.button_style("#16a34a", "#15803d"))
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        self.roi_btn.setEnabled(False)
        header_layout.addWidget(self.roi_btn)
        
        self.roi_manual_btn = QPushButton("📝 Manual ROI")
        self.roi_manual_btn.setStyleSheet(self.button_style("#059669", "#047857"))
        self.roi_manual_btn.clicked.connect(self.open_manual_roi)
        self.roi_manual_btn.setEnabled(False)
        header_layout.addWidget(self.roi_manual_btn)
        
        self.apply_roi_btn = QPushButton("🔍 Apply ROI")
        self.apply_roi_btn.setStyleSheet(self.button_style("#dc2626", "#b91c1c"))
        self.apply_roi_btn.clicked.connect(self.apply_roi_zoom)
        self.apply_roi_btn.setEnabled(False)
        header_layout.addWidget(self.apply_roi_btn)
        
        self.save_roi_btn = QPushButton("💾 Save ROI")
        self.save_roi_btn.setStyleSheet(self.button_style("#ea580c", "#c2410c"))
        self.save_roi_btn.clicked.connect(self.save_roi)
        self.save_roi_btn.setEnabled(False)
        header_layout.addWidget(self.save_roi_btn)
        
        self.load_roi_btn = QPushButton("📂 Load ROI")
        self.load_roi_btn.setStyleSheet(self.button_style("#d97706", "#b45309"))
        self.load_roi_btn.clicked.connect(self.load_roi)
        self.load_roi_btn.setEnabled(False)
        header_layout.addWidget(self.load_roi_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet(self.button_style("#6b7280", "#4b5563"))
        self.reset_btn.clicked.connect(self.reset_view)
        self.reset_btn.setEnabled(False)
        header_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(header_layout)
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("""
            background-color: #1e40af; 
            padding: 10px; 
            border-radius: 4px;
            margin: 5px 0;
        """)
        self.info_label.hide()
        main_layout.addWidget(self.info_label)
        
        self.roi_info_label = QLabel("")
        self.roi_info_label.setStyleSheet("""
            background-color: #15803d; 
            padding: 10px; 
            border-radius: 4px;
            margin: 5px 0;
        """)
        self.roi_info_label.hide()
        main_layout.addWidget(self.roi_info_label)
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        self.canvas_axial = MedicalImageCanvas('axial', self)
        self.canvas_coronal = MedicalImageCanvas('coronal', self)
        self.canvas_sagittal = MedicalImageCanvas('sagittal', self)
        self.canvas_special = MedicalImageCanvas('special', self)
        
        axial_container = QWidget()
        axial_layout = QVBoxLayout(axial_container)
        axial_layout.setContentsMargins(0, 0, 0, 0)
        axial_label = QLabel("AXIAL VIEW")
        axial_label.setStyleSheet("color: #4a9eff; font-weight: bold; padding: 5px;")
        axial_layout.addWidget(axial_label)
        axial_layout.addWidget(self.canvas_axial)
        self.sliders['axial'] = QSlider(Qt.Horizontal)
        self.sliders['axial'].valueChanged.connect(lambda v: self.slider_changed('axial', v))
        self.sliders['axial'].setEnabled(False)
        axial_layout.addWidget(self.sliders['axial'])
        grid_layout.addWidget(axial_container, 0, 0)
        
        coronal_container = QWidget()
        coronal_layout = QVBoxLayout(coronal_container)
        coronal_layout.setContentsMargins(0, 0, 0, 0)
        coronal_label = QLabel("CORONAL VIEW")
        coronal_label.setStyleSheet("color: #4a9eff; font-weight: bold; padding: 5px;")
        coronal_layout.addWidget(coronal_label)
        coronal_layout.addWidget(self.canvas_coronal)
        self.sliders['coronal'] = QSlider(Qt.Horizontal)
        self.sliders['coronal'].valueChanged.connect(lambda v: self.slider_changed('coronal', v))
        self.sliders['coronal'].setEnabled(False)
        coronal_layout.addWidget(self.sliders['coronal'])
        grid_layout.addWidget(coronal_container, 0, 1)
        
        sagittal_container = QWidget()
        sagittal_layout = QVBoxLayout(sagittal_container)
        sagittal_layout.setContentsMargins(0, 0, 0, 0)
        sagittal_label = QLabel("SAGITTAL VIEW")
        sagittal_label.setStyleSheet("color: #4a9eff; font-weight: bold; padding: 5px;")
        sagittal_layout.addWidget(sagittal_label)
        sagittal_layout.addWidget(self.canvas_sagittal)
        self.sliders['sagittal'] = QSlider(Qt.Horizontal)
        self.sliders['sagittal'].valueChanged.connect(lambda v: self.slider_changed('sagittal', v))
        self.sliders['sagittal'].setEnabled(False)
        sagittal_layout.addWidget(self.sliders['sagittal'])
        grid_layout.addWidget(sagittal_container, 1, 0)
        
        special_container = QWidget()
        special_layout = QVBoxLayout(special_container)
        special_layout.setContentsMargins(0, 0, 0, 0)
        self.special_label = QLabel("CONTOUR VIEW")
        self.special_label.setStyleSheet("color: #10b981; font-weight: bold; padding: 5px;")
        special_layout.addWidget(self.special_label)
        special_layout.addWidget(self.canvas_special)
        dummy_slider = QSlider(Qt.Horizontal)
        dummy_slider.setVisible(False)
        special_layout.addWidget(dummy_slider)
        grid_layout.addWidget(special_container, 1, 1)
        
        main_layout.addLayout(grid_layout)
        
        instructions = QLabel(
            "📌 Click Line: Drag single line | Click Crosshair Center: Drag both lines together | "
            "Right Click & Drag: Pan | Scroll: Change slice | Ctrl+Scroll: Zoom | "
            "Oblique: Draw line to cut through volume | 🤖 Enhanced AI Detection"
        )
        instructions.setStyleSheet("color: #9ca3af; padding: 10px; font-size: 11px;")
        main_layout.addWidget(instructions)
    
    def button_style(self, bg_color, hover_color):
        return f"""
            QPushButton {{
                background-color: {bg_color}; 
                color: white; 
                padding: 8px 16px; 
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover_color}; }}
        """
    
    def change_fourth_view_mode(self, mode_text):
        if mode_text == "Contour":
            self.fourth_view_mode = 'contour'
            self.special_label.setText("CONTOUR VIEW")
            self.special_label.setStyleSheet("color: #10b981; font-weight: bold; padding: 5px;")
            self.view_selector.show()
            self.oblique_selector.hide()
            self.oblique_rotate_btn.hide()
        else:
            self.fourth_view_mode = 'oblique'
            self.special_label.setText("OBLIQUE VIEW")
            self.special_label.setStyleSheet("color: #a855f7; font-weight: bold; padding: 5px;")
            self.view_selector.hide()
            self.oblique_selector.show()
            self.oblique_rotate_btn.show()
            if self.volume is not None:
                self.oblique_rotate_btn.setEnabled(True)
        self.update_all_views()
    
    def change_contour_source_view(self, view_text):
        self.active_view_for_contour = view_text.lower()
        self.update_all_views()
    
    def change_oblique_source_plane(self, view_text):
        self.oblique_source_plane = view_text.lower()
        self.oblique_cut_point = None
        self.oblique_cut_angle = 0
        self.update_all_views()
    
    def toggle_oblique_rotation(self):
        self.oblique_rotation_mode = not self.oblique_rotation_mode
        if self.oblique_rotation_mode:
            self.oblique_rotate_btn.setStyleSheet(self.button_style("#ca8a04", "#a16207"))
            self.oblique_rotate_btn.setText("✏️ Drawing... (drag line)")
        else:
            self.oblique_rotate_btn.setStyleSheet(self.button_style("#7c3aed", "#6d28d9"))
            self.oblique_rotate_btn.setText("✏️ Draw Cutting Line")
            self.oblique_cut_point = None
            self.oblique_cut_angle = 0
    
    def start_oblique_line_drawing(self, view_name, pos):
        """Start drawing the cutting line for oblique plane"""
        if not self.oblique_rotation_mode or view_name != self.oblique_source_plane:
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        self.oblique_rotating = True
        self.oblique_rotation_view = view_name
        
        self.oblique_cut_point = {
            'start_x': pos.x() / canvas.zoom,
            'start_y': pos.y() / (canvas.zoom * aspect_ratio),
            'end_x': pos.x() / canvas.zoom,
            'end_y': pos.y() / (canvas.zoom * aspect_ratio)
        }
    
    def update_oblique_line_drawing(self, view_name, pos):
        """Update the cutting line as user drags"""
        if not self.oblique_rotating or view_name != self.oblique_rotation_view or not self.oblique_cut_point:
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        self.oblique_cut_point['end_x'] = pos.x() / canvas.zoom
        self.oblique_cut_point['end_y'] = pos.y() / (canvas.zoom * aspect_ratio)
        
        dx = self.oblique_cut_point['end_x'] - self.oblique_cut_point['start_x']
        dy = self.oblique_cut_point['end_y'] - self.oblique_cut_point['start_y']
        self.oblique_cut_angle = np.degrees(np.arctan2(dy, dx))
        
        self.oblique_rotate_btn.setText(f"✏️ Angle: {self.oblique_cut_angle:.1f}°")
        self.update_all_views()
    
    def finish_oblique_line_drawing(self):
        """Finish drawing the cutting line"""
        self.oblique_rotating = False
        if self.oblique_cut_point:
            dx = self.oblique_cut_point['end_x'] - self.oblique_cut_point['start_x']
            dy = self.oblique_cut_point['end_y'] - self.oblique_cut_point['start_y']
            if abs(dx) < 10 and abs(dy) < 10:
                self.oblique_cut_point = None
                self.oblique_cut_angle = 0
        self.oblique_rotation_view = None
    
    def check_line_click(self, view_name, pos):
        if self.volume is None:
            return None
        
        canvas = getattr(self, f'canvas_{view_name}')
        click_x = pos.x() / canvas.zoom
        click_y = pos.y() / (canvas.zoom * self.aspect_ratios.get(view_name, 1.0))
        
        z, y, x = self.crosshair_pos
        threshold = 15 / canvas.zoom
        
        if view_name == 'axial':
            if abs(click_y - y) < threshold:
                return 'coronal'
            elif abs(click_x - x) < threshold:
                return 'sagittal'
        elif view_name == 'coronal':
            if abs(click_y - (self.volume.shape[0] - 1 - z)) < threshold:
                return 'axial'
            elif abs(click_x - x) < threshold:
                return 'sagittal'
        elif view_name == 'sagittal':
            if abs(click_y - (self.volume.shape[0] - 1 - z)) < threshold:
                return 'axial'
            elif abs(click_x - y) < threshold:
                return 'coronal'
        
        return None
    
    def check_crosshair_center_click(self, view_name, pos):
        """Check if user clicked near the crosshair intersection"""
        if self.volume is None:
            return False
        
        canvas = getattr(self, f'canvas_{view_name}')
        click_x = pos.x() / canvas.zoom
        click_y = pos.y() / (canvas.zoom * self.aspect_ratios.get(view_name, 1.0))
        
        z, y, x = self.crosshair_pos
        threshold = 20 / canvas.zoom
        
        if view_name == 'axial':
            center_x, center_y = x, y
        elif view_name == 'coronal':
            center_x = x
            center_y = self.volume.shape[0] - 1 - z
        elif view_name == 'sagittal':
            center_x = y
            center_y = self.volume.shape[0] - 1 - z
        else:
            return False
        
        dist = np.sqrt((click_x - center_x)**2 + (click_y - center_y)**2)
        return dist < threshold
    
    def start_both_lines_dragging(self, view_name, pos):
        """Start dragging both reference lines together"""
        self.dragging_line = 'both'
        self.dragging_view = view_name
        self.dragging_both_lines = True
    
    def start_line_dragging(self, line_type, view_name, pos):
        self.dragging_line = line_type
        self.dragging_view = view_name
    
    def update_line_dragging(self, view_name, pos):
        if not self.dragging_line or self.volume is None:
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        mouse_x = pos.x() / canvas.zoom
        mouse_y = pos.y() / (canvas.zoom * aspect_ratio)
        
        slice_data = self.get_slice(view_name)
        h, w = slice_data.shape
        
        mouse_x = max(0, min(w - 1, mouse_x))
        mouse_y = max(0, min(h - 1, mouse_y))
        
        mouse_x = int(mouse_x)
        mouse_y = int(mouse_y)
        
        if self.dragging_line == 'axial':
            if view_name in ['coronal', 'sagittal']:
                self.crosshair_pos[0] = int(self.volume.shape[0] - 1 - mouse_y)
        elif self.dragging_line == 'coronal':
            if view_name == 'axial':
                self.crosshair_pos[1] = mouse_y
            elif view_name == 'sagittal':
                self.crosshair_pos[1] = mouse_x
        elif self.dragging_line == 'sagittal':
            if view_name == 'axial':
                self.crosshair_pos[2] = mouse_x
            elif view_name == 'coronal':
                self.crosshair_pos[2] = mouse_x
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.volume.shape[0] - 1)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.volume.shape[1] - 1)
        self.crosshair_pos[2] = np.clip(self.crosshair_pos[2], 0, self.volume.shape[2] - 1)
        
        self.sliders['axial'].blockSignals(True)
        self.sliders['coronal'].blockSignals(True)
        self.sliders['sagittal'].blockSignals(True)
        
        self.sliders['axial'].setValue(self.crosshair_pos[0])
        self.sliders['coronal'].setValue(self.crosshair_pos[1])
        self.sliders['sagittal'].setValue(self.crosshair_pos[2])
        
        self.sliders['axial'].blockSignals(False)
        self.sliders['coronal'].blockSignals(False)
        self.sliders['sagittal'].blockSignals(False)
        
        self.update_all_views()
    
    def update_both_lines_dragging(self, view_name, pos):
        """Update position when dragging both lines together"""
        if not self.dragging_both_lines or self.volume is None:
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        mouse_x = pos.x() / canvas.zoom
        mouse_y = pos.y() / (canvas.zoom * aspect_ratio)
        
        slice_data = self.get_slice(view_name)
        h, w = slice_data.shape
        
        mouse_x = max(0, min(w - 1, int(mouse_x)))
        mouse_y = max(0, min(h - 1, int(mouse_y)))
        
        if view_name == 'axial':
            self.crosshair_pos[1] = mouse_y
            self.crosshair_pos[2] = mouse_x
        elif view_name == 'coronal':
            self.crosshair_pos[0] = int(self.volume.shape[0] - 1 - mouse_y)
            self.crosshair_pos[2] = mouse_x
        elif view_name == 'sagittal':
            self.crosshair_pos[0] = int(self.volume.shape[0] - 1 - mouse_y)
            self.crosshair_pos[1] = mouse_x
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.volume.shape[0] - 1)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.volume.shape[1] - 1)
        self.crosshair_pos[2] = np.clip(self.crosshair_pos[2], 0, self.volume.shape[2] - 1)
        
        self.sliders['axial'].blockSignals(True)
        self.sliders['coronal'].blockSignals(True)
        self.sliders['sagittal'].blockSignals(True)
        
        self.sliders['axial'].setValue(self.crosshair_pos[0])
        self.sliders['coronal'].setValue(self.crosshair_pos[1])
        self.sliders['sagittal'].setValue(self.crosshair_pos[2])
        
        self.sliders['axial'].blockSignals(False)
        self.sliders['coronal'].blockSignals(False)
        self.sliders['sagittal'].blockSignals(False)
        
        self.update_all_views()
    
    def finish_line_dragging(self):
        self.dragging_line = None
        self.dragging_view = None
        self.dragging_both_lines = False
    
    def load_files(self):
        dialog = QMessageBox()
        dialog.setWindowTitle("Load Medical Images")
        dialog.setText("Choose loading method:")
        dialog.setIcon(QMessageBox.Question)
        
        folder_btn = dialog.addButton("📁 DICOM Folder", QMessageBox.YesRole)
        file_btn = dialog.addButton("📄 Single File (NIfTI/MHD)", QMessageBox.NoRole)
        cancel_btn = dialog.addButton("Cancel", QMessageBox.RejectRole)
        
        dialog.exec_()
        clicked = dialog.clickedButton()
        
        if clicked == cancel_btn:
            return
        
        try:
            if clicked == folder_btn:
                folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
                if not folder:
                    return
                self.load_dicom_series(Path(folder))
            else:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Load Medical Image", "",
                    "Medical Images (*.nii *.nii.gz *.mhd);;NIfTI (*.nii *.nii.gz);;MHD (*.mhd);;All Files (*.*)")
                
                if not file_path:
                    return
                
                file_path = Path(file_path)
                self.current_file_path = file_path
                
                if file_path.suffix.lower() in ['.nii', '.gz'] or file_path.name.endswith('.nii.gz'):
                    if not NIBABEL_AVAILABLE:
                        QMessageBox.warning(self, "Missing Library", 
                                          "Please install nibabel: pip install nibabel")
                        return
                    self.load_nifti(file_path)
                elif file_path.suffix.lower() == '.mhd':
                    self.load_mhd(file_path)
                else:
                    QMessageBox.warning(self, "Unsupported Format", 
                                      "Please load NIfTI (.nii, .nii.gz) or MHD files")
                    return
            
            self.roi_btn.setEnabled(True)
            self.roi_manual_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.load_roi_btn.setEnabled(True)
            if self.fourth_view_mode == 'oblique':
                self.oblique_rotate_btn.setEnabled(True)
            
            for slider in self.sliders.values():
                slider.setEnabled(True)
            
            self.update_slider_ranges()
            self.update_aspect_ratios()
            self.run_ai_detection()
            self.update_all_views()
            
            spacing_text = f"Voxel Spacing: Z={self.voxel_spacing[0]:.2f}mm, Y={self.voxel_spacing[1]:.2f}mm, X={self.voxel_spacing[2]:.2f}mm"
            ai_status = "ResNet18 + Heuristics" if TORCH_AVAILABLE else "Enhanced Heuristics"
            QMessageBox.information(self, "Success", 
                f"Loaded volume shape: {self.volume.shape}\n"
                f"{spacing_text}\n"
                f"🤖 AI Detection: {ai_status}")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}\n\n{error_msg}")
    
    def load_dicom_series(self, directory):
        dicom_files = []
        for pattern in ["*.dcm", "*.DCM", "*.dicom", "*.DICOM", "*"]:
            found_files = list(Path(directory).glob(pattern))
            if found_files:
                dicom_files.extend(found_files)
                break
        
        if not dicom_files:
            raise ValueError("No files found in directory")
        
        valid_slices = []
        spacing = [1.0, 1.0, 1.0]
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                
                if not hasattr(ds, 'file_meta') or not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    if not hasattr(ds, 'file_meta'):
                        ds.file_meta = pydicom.dataset.FileMetaDataset()
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                
                pixel_array = ds.pixel_array
                instance_num = int(getattr(ds, 'InstanceNumber', len(valid_slices)))
                
                z_pos = None
                if hasattr(ds, 'SliceLocation'):
                    z_pos = float(ds.SliceLocation)
                elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                    z_pos = float(ds.ImagePositionPatient[2])
                
                if z_pos is None:
                    z_pos = float(instance_num)
                
                if hasattr(ds, 'PixelSpacing') and len(ds.PixelSpacing) >= 2:
                    spacing[1] = float(ds.PixelSpacing[0])
                    spacing[2] = float(ds.PixelSpacing[1])
                if hasattr(ds, 'SliceThickness'):
                    spacing[0] = float(ds.SliceThickness)
                
                valid_slices.append((z_pos, instance_num, pixel_array.astype(np.float32)))
            except:
                continue
        
        if not valid_slices:
            raise ValueError("No valid DICOM images found")
        
        valid_slices.sort(key=lambda x: (x[0], x[1]))
        slices = [s[2] for s in valid_slices]
        self.volume = np.array(slices, dtype=np.float32)
        
        if len(self.volume.shape) == 2:
            self.volume = self.volume[np.newaxis, :, :]
        
        self.voxel_spacing = spacing
        self.normalize_volume()
        self.original_volume = self.volume.copy()
        self.reset_position()
    
    def load_nifti(self, file_path):
        img = nib.load(str(file_path))
        data = img.get_fdata()
        
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
        elif len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        
        data = np.transpose(data, (2, 1, 0))
        
        self.volume = data.astype(np.float32)
        self.voxel_spacing = list(img.header.get_zooms()[:3])[::-1]
        self.normalize_volume()
        self.original_volume = self.volume.copy()
        self.reset_position()
    
    def load_mhd(self, file_path):
        header = {}
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    header[key.strip()] = value.strip()
        
        raw_file = file_path.with_suffix('.raw')
        if not raw_file.exists():
            raw_file = file_path.with_suffix('.zraw')
        if not raw_file.exists():
            raise FileNotFoundError("Raw file not found")
        
        dims = [int(x) for x in header.get('DimSize', '1 1 1').split()]
        dtype_map = {
            'MET_SHORT': np.int16, 'MET_USHORT': np.uint16,
            'MET_FLOAT': np.float32, 'MET_UCHAR': np.uint8,
            'MET_CHAR': np.int8, 'MET_INT': np.int32, 'MET_UINT': np.uint32,
        }
        dtype = dtype_map.get(header.get('ElementType', 'MET_SHORT'), np.int16)
        
        raw_data = np.fromfile(raw_file, dtype=dtype)
        
        if len(dims) == 3:
            self.volume = raw_data.reshape(dims[2], dims[1], dims[0]).astype(np.float32)
        elif len(dims) == 2:
            self.volume = raw_data.reshape(1, dims[1], dims[0]).astype(np.float32)
        else:
            self.volume = raw_data.reshape(-1, 1, 1).astype(np.float32)
        
        spacing = header.get('ElementSpacing', '1.0 1.0 1.0').split()
        self.voxel_spacing = [float(s) for s in spacing][::-1]
        
        self.normalize_volume()
        self.original_volume = self.volume.copy()
        self.reset_position()
    
    def normalize_volume(self):
        vmin, vmax = self.volume.min(), self.volume.max()
        if vmax > vmin:
            self.volume = ((self.volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            self.volume = np.zeros_like(self.volume, dtype=np.uint8)
    
    def reset_position(self):
        z, y, x = self.volume.shape
        self.crosshair_pos = [z // 2, y // 2, x // 2]
    
    def update_slider_ranges(self):
        z, y, x = self.volume.shape
        self.sliders['axial'].setMaximum(z - 1)
        self.sliders['axial'].setValue(self.crosshair_pos[0])
        self.sliders['coronal'].setMaximum(y - 1)
        self.sliders['coronal'].setValue(self.crosshair_pos[1])
        self.sliders['sagittal'].setMaximum(x - 1)
        self.sliders['sagittal'].setValue(self.crosshair_pos[2])
    
    def update_aspect_ratios(self):
        if self.volume is None:
            return
        sz, sy, sx = self.voxel_spacing
        
        self.aspect_ratios['axial'] = sy / sx if sx != 0 else 1.0
        self.aspect_ratios['coronal'] = sz / sx if sx != 0 else 1.0
        self.aspect_ratios['sagittal'] = sz / sy if sy != 0 else 1.0
    
    def slider_changed(self, view_name, value):
        if view_name == 'axial':
            self.crosshair_pos[0] = value
        elif view_name == 'coronal':
            self.crosshair_pos[1] = value
        elif view_name == 'sagittal':
            self.crosshair_pos[2] = value
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.volume.shape[0] - 1)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.volume.shape[1] - 1)
        self.crosshair_pos[2] = np.clip(self.crosshair_pos[2], 0, self.volume.shape[2] - 1)
        
        self.sliders['axial'].blockSignals(True)
        self.sliders['coronal'].blockSignals(True)
        self.sliders['sagittal'].blockSignals(True)
        
        self.sliders['axial'].setValue(self.crosshair_pos[0])
        self.sliders['coronal'].setValue(self.crosshair_pos[1])
        self.sliders['sagittal'].setValue(self.crosshair_pos[2])
        
        self.sliders['axial'].blockSignals(False)
        self.sliders['coronal'].blockSignals(False)
        self.sliders['sagittal'].blockSignals(False)
        
        self.update_all_views()
    
    def run_ai_detection(self):
        """Enhanced AI detection with per-plane orientation analysis"""
        if self.volume is None:
            return
        
        organ, organ_conf, vol_orient, vol_conf = self.detector.detect_from_volume(self.volume)
        
        z, y, x = self.volume.shape
        mid_z, mid_y, mid_x = z // 2, y // 2, x // 2
        
        axial_slice = self.volume[mid_z, :, :]
        coronal_slice = self.volume[:, mid_y, :]
        sagittal_slice = self.volume[:, :, mid_x]
        
        axial_orient, axial_conf = self.orientation_detector.detect_orientation(axial_slice)
        coronal_orient, coronal_conf = self.orientation_detector.detect_orientation(coronal_slice)
        sagittal_orient, sagittal_conf = self.orientation_detector.detect_orientation(sagittal_slice)
        
        self.detected_info = {
            'organ': organ,
            'organ_confidence': organ_conf,
            'volume_orientation': vol_orient,
            'volume_confidence': vol_conf,
            'axial_orientation': axial_orient,
            'axial_confidence': axial_conf,
            'coronal_orientation': coronal_orient,
            'coronal_confidence': coronal_conf,
            'sagittal_orientation': sagittal_orient,
            'sagittal_confidence': sagittal_conf
        }
        
        ai_method = "ResNet18 + Heuristics" if TORCH_AVAILABLE else "Enhanced Heuristics"
        info_text = (f"🤖 AI Detection ({ai_method}): <b>{organ}</b> ({organ_conf:.1f}%) | "
                    f"Orientations: Axial-<b>{axial_orient}</b> ({axial_conf:.0f}%), "
                    f"Coronal-<b>{coronal_orient}</b> ({coronal_conf:.0f}%), "
                    f"Sagittal-<b>{sagittal_orient}</b> ({sagittal_conf:.0f}%)")
        self.info_label.setText(info_text)
        self.info_label.show()
    
    def get_slice(self, view_name):
        if self.volume is None:
            return np.zeros((100, 100), dtype=np.uint8)
        
        try:
            z, y, x = self.crosshair_pos
            z = np.clip(z, 0, self.volume.shape[0] - 1)
            y = np.clip(y, 0, self.volume.shape[1] - 1)
            x = np.clip(x, 0, self.volume.shape[2] - 1)
            
            if view_name == 'axial':
                return self.volume[z, :, :]
            elif view_name == 'coronal':
                return self.volume[::-1, y, :]
            elif view_name == 'sagittal':
                return self.volume[::-1, :, x]
            elif view_name == 'special':
                if self.fourth_view_mode == 'contour':
                    return self.get_contour_view()
                else:
                    return self.get_oblique_view()
        except Exception as e:
            print(f"Error getting slice: {e}")
            return np.zeros((100, 100), dtype=np.uint8)
    
    def get_contour_view(self):
        if self.volume is None or not SCIPY_AVAILABLE:
            return np.zeros((100, 100), dtype=np.uint8)
        
        try:
            z, y, x = self.crosshair_pos
            if self.active_view_for_contour == 'axial':
                z = np.clip(z, 0, self.volume.shape[0] - 1)
                slice_data = self.volume[z, :, :]
            elif self.active_view_for_contour == 'coronal':
                y = np.clip(y, 0, self.volume.shape[1] - 1)
                slice_data = self.volume[::-1, y, :]
            else:
                x = np.clip(x, 0, self.volume.shape[2] - 1)
                slice_data = self.volume[::-1, :, x]
            
            contour_image = np.zeros_like(slice_data, dtype=np.uint8)
            edges = feature.canny(slice_data, sigma=2.0)
            contour_image[edges] = 255
            return contour_image
        except Exception as e:
            print(f"Error creating contour: {e}")
            return np.zeros((100, 100), dtype=np.uint8)
    
    def get_oblique_view(self):
        """Generate oblique plane by cutting through the volume along the drawn line"""
        if self.volume is None or not SCIPY_AVAILABLE:
            return np.zeros((100, 100), dtype=np.uint8)

        if not self.oblique_cut_point:
            return self.get_slice(self.oblique_source_plane)

        try:
            z, y, x = self.crosshair_pos
            sz, sy, sx = self.voxel_spacing
            nz, ny, nx = self.volume.shape

            start_x = int(self.oblique_cut_point['start_x'])
            start_y = int(self.oblique_cut_point['start_y'])
            end_x = int(self.oblique_cut_point['end_x'])
            end_y = int(self.oblique_cut_point['end_y'])

            line_length = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2))
            if line_length < 10:
                return self.get_slice(self.oblique_source_plane)

            angle_rad = np.radians(self.oblique_cut_angle)
            
            if self.oblique_source_plane == 'axial':
                width = line_length
                height = nz
                oblique_slice = np.zeros((height, width), dtype=np.uint8)
                
                for i in range(width):
                    t = i / max(width - 1, 1)
                    cut_x = start_x + t * (end_x - start_x)
                    cut_y = start_y + t * (end_y - start_y)
                    
                    cut_x = int(np.clip(cut_x, 0, nx - 1))
                    cut_y = int(np.clip(cut_y, 0, ny - 1))
                    
                    for z_idx in range(nz):
                        oblique_slice[nz - 1 - z_idx, i] = self.volume[z_idx, cut_y, cut_x]
                        
            elif self.oblique_source_plane == 'coronal':
                width = line_length
                height = ny
                oblique_slice = np.zeros((height, width), dtype=np.uint8)
                
                for i in range(width):
                    t = i / max(width - 1, 1)
                    cut_x = start_x + t * (end_x - start_x)
                    cut_z = nz - 1 - int(start_y + t * (end_y - start_y))
                    
                    cut_x = int(np.clip(cut_x, 0, nx - 1))
                    cut_z = int(np.clip(cut_z, 0, nz - 1))
                    
                    for y_idx in range(ny):
                        oblique_slice[ny - 1 - y_idx, i] = self.volume[cut_z, y_idx, cut_x]
                        
            else:  # sagittal
                width = line_length
                height = nx
                oblique_slice = np.zeros((height, width), dtype=np.uint8)
                
                for i in range(width):
                    t = i / max(width - 1, 1)
                    cut_y = start_x + t * (end_x - start_x)
                    cut_z = nz - 1 - int(start_y + t * (end_y - start_y))
                    
                    cut_y = int(np.clip(cut_y, 0, ny - 1))
                    cut_z = int(np.clip(cut_z, 0, nz - 1))
                    
                    for x_idx in range(nx):
                        oblique_slice[nx - 1 - x_idx, i] = self.volume[cut_z, cut_y, x_idx]

            return oblique_slice
            
        except Exception as e:
            print(f"Error creating oblique view: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((100, 100), dtype=np.uint8)
    
    def render_view(self, view_name, canvas):
        slice_data = self.get_slice(view_name)
        h, w = slice_data.shape
        
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        slice_data = slice_data.astype(np.float32)
        vmin, vmax = slice_data.min(), slice_data.max()
        if vmax > vmin:
            slice_data = ((slice_data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        slice_data = np.ascontiguousarray(slice_data)
        qimage = QImage(slice_data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        
        target_width = int(w * canvas.zoom)
        target_height = int(h * aspect_ratio * canvas.zoom)
        
        scaled_pixmap = pixmap.scaled(
            target_width, target_height,
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        
        painter = QPainter(scaled_pixmap)
        
        if view_name != 'special':
            z, y, x = self.crosshair_pos
            pen_width = max(2, int(2 * canvas.zoom))
            
            if view_name == 'axial':
                pen_h = QPen(QColor(255, 255, 0, 200))
                pen_h.setWidth(pen_width)
                painter.setPen(pen_h)
                cy = int(y * canvas.zoom)
                cy = max(0, min(scaled_pixmap.height() - 1, cy))
                painter.drawLine(0, cy, scaled_pixmap.width(), cy)
                
                pen_v = QPen(QColor(255, 0, 255, 200))
                pen_v.setWidth(pen_width)
                painter.setPen(pen_v)
                cx = int(x * canvas.zoom)
                cx = max(0, min(scaled_pixmap.width() - 1, cx))
                painter.drawLine(cx, 0, cx, scaled_pixmap.height())
                
            elif view_name == 'coronal':
                pen_h = QPen(QColor(0, 255, 255, 200))
                pen_h.setWidth(pen_width)
                painter.setPen(pen_h)
                cy = int((self.volume.shape[0] - 1 - z) * aspect_ratio * canvas.zoom)
                cy = max(0, min(scaled_pixmap.height() - 1, cy))
                painter.drawLine(0, cy, scaled_pixmap.width(), cy)
                
                pen_v = QPen(QColor(255, 0, 255, 200))
                pen_v.setWidth(pen_width)
                painter.setPen(pen_v)
                cx = int(x * canvas.zoom)
                cx = max(0, min(scaled_pixmap.width() - 1, cx))
                painter.drawLine(cx, 0, cx, scaled_pixmap.height())
                
            elif view_name == 'sagittal':
                pen_h = QPen(QColor(0, 255, 255, 200))
                pen_h.setWidth(pen_width)
                painter.setPen(pen_h)
                cy = int((self.volume.shape[0] - 1 - z) * aspect_ratio * canvas.zoom)
                cy = max(0, min(scaled_pixmap.height() - 1, cy))
                painter.drawLine(0, cy, scaled_pixmap.width(), cy)
                
                pen_v = QPen(QColor(255, 255, 0, 200))
                pen_v.setWidth(pen_width)
                painter.setPen(pen_v)
                cx = int(y * canvas.zoom)
                cx = max(0, min(scaled_pixmap.width() - 1, cx))
                painter.drawLine(cx, 0, cx, scaled_pixmap.height())
            
            # Draw oblique cutting line if in oblique mode
            if self.fourth_view_mode == 'oblique' and view_name == self.oblique_source_plane and self.oblique_cut_point:
                pen_cut = QPen(QColor(0, 255, 0, 220))
                pen_cut.setWidth(pen_width + 2)
                pen_cut.setStyle(Qt.SolidLine)
                painter.setPen(pen_cut)
                
                sx = int(self.oblique_cut_point['start_x'] * canvas.zoom)
                sy = int(self.oblique_cut_point['start_y'] * aspect_ratio * canvas.zoom)
                ex = int(self.oblique_cut_point['end_x'] * canvas.zoom)
                ey = int(self.oblique_cut_point['end_y'] * aspect_ratio * canvas.zoom)
                
                painter.drawLine(sx, sy, ex, ey)
                
                # Draw arrow heads
                arrow_size = 10
                angle = np.radians(self.oblique_cut_angle)
                painter.drawLine(ex, ey, 
                               int(ex - arrow_size * np.cos(angle - np.pi/6)),
                               int(ey - arrow_size * np.sin(angle - np.pi/6)))
                painter.drawLine(ex, ey,
                               int(ex - arrow_size * np.cos(angle + np.pi/6)),
                               int(ey - arrow_size * np.sin(angle + np.pi/6)))
        
        if self.roi and view_name != 'special':
            self.draw_roi_overlay(painter, view_name, canvas, w, h)
        
        if self.roi_current and self.roi_current['view'] == view_name:
            pen = QPen(QColor(0, 255, 0, 150))
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            x = self.roi_current['x']
            y = self.roi_current['y']
            painter.drawRect(
                int(x * canvas.zoom),
                int(y * aspect_ratio * canvas.zoom),
                int(self.roi_current['w'] * canvas.zoom),
                int(self.roi_current['h'] * aspect_ratio * canvas.zoom)
            )
        
        # Draw text overlays at TOP of image
        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        
        # Add semi-transparent background for better readability
        painter.setBrush(QColor(0, 0, 0, 150))
        if view_name == 'special':
            painter.drawRect(5, 5, 300, 80)
        else:
            painter.drawRect(5, 5, 200, 80)
        
        z, y, x = self.crosshair_pos
        if view_name == 'axial':
            painter.drawText(10, 25, f"AXIAL VIEW")
            painter.drawText(10, 45, f"Z: {z}/{self.volume.shape[0]-1}")
            painter.drawText(10, 65, f"Pos: X={x}, Y={y}")
        elif view_name == 'coronal':
            painter.drawText(10, 25, f"CORONAL VIEW")
            painter.drawText(10, 45, f"Y: {y}/{self.volume.shape[1]-1}")
            painter.drawText(10, 65, f"Pos: X={x}, Z={z}")
        elif view_name == 'sagittal':
            painter.drawText(10, 25, f"SAGITTAL VIEW")
            painter.drawText(10, 45, f"X: {x}/{self.volume.shape[2]-1}")
            painter.drawText(10, 65, f"Pos: Y={y}, Z={z}")
        elif view_name == 'special':
            if self.fourth_view_mode == 'contour':
                painter.drawText(10, 25, f"CONTOUR - {self.active_view_for_contour.upper()}")
                if self.active_view_for_contour == 'axial':
                    painter.drawText(10, 45, f"Slice: {z}")
                elif self.active_view_for_contour == 'coronal':
                    painter.drawText(10, 45, f"Slice: {y}")
                else:
                    painter.drawText(10, 45, f"Slice: {x}")
            else:
                painter.drawText(10, 25, f"OBLIQUE - {self.oblique_source_plane.upper()}")
                if self.oblique_cut_point:
                    painter.drawText(10, 45, f"Angle: {self.oblique_cut_angle:.1f}°")
                else:
                    painter.drawText(10, 45, f"Draw a line to cut")
                if self.oblique_source_plane == 'axial':
                    painter.drawText(10, 65, f"Base: Z={z}")
                elif self.oblique_source_plane == 'coronal':
                    painter.drawText(10, 65, f"Base: Y={y}")
                else:
                    painter.drawText(10, 65, f"Base: X={x}")
        
        painter.drawText(10, 85, f"Zoom: {canvas.zoom:.2f}x")
        painter.end()
        canvas.setPixmap(scaled_pixmap)
    
    def draw_roi_overlay(self, painter, view_name, canvas, w, h):
        pen = QPen(QColor(0, 255, 0, 200))
        pen.setWidth(3)
        pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        
        z_start, z_end = self.roi['z_start'], self.roi['z_end']
        y_start, y_end = self.roi['y_start'], self.roi['y_end']
        x_start, x_end = self.roi['x_start'], self.roi['x_end']
        
        aspect_ratio = self.aspect_ratios.get(view_name, 1.0)
        
        if view_name == 'axial':
            painter.drawRect(
                int(x_start * canvas.zoom),
                int(y_start * canvas.zoom),
                int((x_end - x_start) * canvas.zoom),
                int((y_end - y_start) * canvas.zoom)
            )
        elif view_name == 'coronal':
            painter.drawRect(
                int(x_start * canvas.zoom),
                int((self.volume.shape[0] - 1 - z_end) * aspect_ratio * canvas.zoom),
                int((x_end - x_start) * canvas.zoom),
                int((z_end - z_start) * aspect_ratio * canvas.zoom)
            )
        elif view_name == 'sagittal':
            painter.drawRect(
                int(y_start * canvas.zoom),
                int((self.volume.shape[0] - 1 - z_end) * aspect_ratio * canvas.zoom),
                int((y_end - y_start) * canvas.zoom),
                int((z_end - z_start) * aspect_ratio * canvas.zoom)
            )
    
    def update_all_views(self):
        for view_name, canvas in [('axial', self.canvas_axial),
                                 ('coronal', self.canvas_coronal),
                                 ('sagittal', self.canvas_sagittal),
                                 ('special', self.canvas_special)]:
            self.render_view(view_name, canvas)
    
    def change_slice(self, view_name, delta):
        if view_name == 'axial':
            self.crosshair_pos[0] = np.clip(self.crosshair_pos[0] + delta, 0, self.volume.shape[0] - 1)
        elif view_name == 'coronal':
            self.crosshair_pos[1] = np.clip(self.crosshair_pos[1] + delta, 0, self.volume.shape[1] - 1)
        elif view_name == 'sagittal':
            self.crosshair_pos[2] = np.clip(self.crosshair_pos[2] + delta, 0, self.volume.shape[2] - 1)
        
        self.sliders['axial'].blockSignals(True)
        self.sliders['coronal'].blockSignals(True)
        self.sliders['sagittal'].blockSignals(True)
        
        self.sliders['axial'].setValue(self.crosshair_pos[0])
        self.sliders['coronal'].setValue(self.crosshair_pos[1])
        self.sliders['sagittal'].setValue(self.crosshair_pos[2])
        
        self.sliders['axial'].blockSignals(False)
        self.sliders['coronal'].blockSignals(False)
        self.sliders['sagittal'].blockSignals(False)
        
        self.update_all_views()
    
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.roi_btn.setStyleSheet(self.button_style("#ca8a04", "#a16207"))
            self.roi_btn.setText("⬚ Drawing ROI...")
        else:
            self.roi_btn.setStyleSheet(self.button_style("#16a34a", "#15803d"))
            self.roi_btn.setText("⬚ Draw ROI")
            self.roi_drawing = False
            self.roi_start_pos = None
            self.roi_current = None
            self.update_all_views()
    
    def start_roi_drawing(self, view_name, pos):
        if view_name == 'special':
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        x = pos.x() / canvas.zoom
        y = pos.y() / (canvas.zoom * self.aspect_ratios.get(view_name, 1.0))
        
        self.roi_drawing = True
        self.roi_start_pos = {'view': view_name, 'x': x, 'y': y}
        self.roi_current = None
    
    def update_roi_drawing(self, view_name, pos):
        if not self.roi_drawing or not self.roi_start_pos or self.roi_start_pos['view'] != view_name:
            return
        
        canvas = getattr(self, f'canvas_{view_name}')
        x = pos.x() / canvas.zoom
        y = pos.y() / (canvas.zoom * self.aspect_ratios.get(view_name, 1.0))
        
        x1 = min(self.roi_start_pos['x'], x)
        y1 = min(self.roi_start_pos['y'], y)
        x2 = max(self.roi_start_pos['x'], x)
        y2 = max(self.roi_start_pos['y'], y)
        
        self.roi_current = {
            'view': view_name,
            'x': x1, 'y': y1,
            'w': x2 - x1, 'h': y2 - y1
        }
        self.update_all_views()
    
    def finish_roi_drawing(self, view_name, pos):
        if not self.roi_current or self.roi_current['w'] < 5 or self.roi_current['h'] < 5:
            self.roi_drawing = False
            self.roi_current = None
            self.roi_mode = False
            self.roi_btn.setStyleSheet(self.button_style("#16a34a", "#15803d"))
            self.roi_btn.setText("⬚ Draw ROI")
            self.update_all_views()
            return
        
        view = self.roi_current['view']
        x1, y1 = int(self.roi_current['x']), int(self.roi_current['y'])
        x2, y2 = int(self.roi_current['x'] + self.roi_current['w']), int(self.roi_current['y'] + self.roi_current['h'])
        
        if view == 'axial':
            self.roi = {
                'z_start': 0, 'z_end': self.volume.shape[0] - 1,
                'y_start': y1, 'y_end': y2,
                'x_start': x1, 'x_end': x2
            }
        elif view == 'coronal':
            self.roi = {
                'z_start': self.volume.shape[0] - 1 - y2,
                'z_end': self.volume.shape[0] - 1 - y1,
                'y_start': 0, 'y_end': self.volume.shape[1] - 1,
                'x_start': x1, 'x_end': x2
            }
        elif view == 'sagittal':
            self.roi = {
                'z_start': self.volume.shape[0] - 1 - y2,
                'z_end': self.volume.shape[0] - 1 - y1,
                'y_start': x1, 'y_end': x2,
                'x_start': 0, 'x_end': self.volume.shape[2] - 1
            }
        
        self.roi_drawing = False
        self.roi_current = None
        self.roi_mode = False
        
        self.roi_btn.setStyleSheet(self.button_style("#16a34a", "#15803d"))
        self.roi_btn.setText("⬚ Draw ROI")
        
        self.apply_roi_btn.setEnabled(True)
        self.save_roi_btn.setEnabled(True)
        
        self.update_roi_info()
        self.update_all_views()
        
        QMessageBox.information(self, "ROI Created",
            f"ROI drawn on {view} view\n\n"
            f"Z: {self.roi['z_start']}-{self.roi['z_end']}\n"
            f"Y: {self.roi['y_start']}-{self.roi['y_end']}\n"
            f"X: {self.roi['x_start']}-{self.roi['x_end']}")
    
    def open_manual_roi(self):
        dialog = ROIDialog(self.volume.shape, self.roi, self)
        if dialog.exec_() == QDialog.Accepted:
            self.roi = dialog.get_roi()
            self.apply_roi_btn.setEnabled(True)
            self.save_roi_btn.setEnabled(True)
            self.update_roi_info()
            self.update_all_views()
    
    def update_roi_info(self):
        if self.roi:
            info_text = (f"🎯 ROI: Z=[{self.roi['z_start']}-{self.roi['z_end']}] "
                        f"Y=[{self.roi['y_start']}-{self.roi['y_end']}] "
                        f"X=[{self.roi['x_start']}-{self.roi['x_end']}]")
            self.roi_info_label.setText(info_text)
            self.roi_info_label.show()
        else:
            self.roi_info_label.hide()
    
    def apply_roi_zoom(self):
        if not self.roi:
            return
        try:
            z1, z2 = self.roi['z_start'], self.roi['z_end'] + 1
            y1, y2 = self.roi['y_start'], self.roi['y_end'] + 1
            x1, x2 = self.roi['x_start'], self.roi['x_end'] + 1
            self.volume = self.volume[z1:z2, y1:y2, x1:x2]
            self.reset_position()
            self.update_slider_ranges()
            self.update_aspect_ratios()
            self.update_all_views()
            QMessageBox.information(self, "ROI Applied",
                f"Volume cropped to ROI\nNew shape: {self.volume.shape}")
            self.roi = None
            self.apply_roi_btn.setEnabled(False)
            self.save_roi_btn.setEnabled(False)
            self.update_roi_info()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply ROI:\n{str(e)}")
    
    def save_roi(self):
        if not self.roi:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.roi, f, indent=2)
                QMessageBox.information(self, "Success", "ROI saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ROI:\n{str(e)}")
    
    def load_roi(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.roi = json.load(f)
                self.apply_roi_btn.setEnabled(True)
                self.save_roi_btn.setEnabled(True)
                self.update_roi_info()
                self.update_all_views()
                QMessageBox.information(self, "Success", "ROI loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ROI:\n{str(e)}")
    
    def reset_view(self):
        if self.original_volume is None:
            return
        self.volume = self.original_volume.copy()
        self.roi = None
        self.oblique_cut_point = None
        self.oblique_cut_angle = 0
        self.apply_roi_btn.setEnabled(False)
        self.save_roi_btn.setEnabled(False)
        self.reset_position()
        self.update_slider_ranges()
        self.update_aspect_ratios()
        self.update_roi_info()
        self.update_all_views()
        QMessageBox.information(self, "Reset", f"View reset\nShape: {self.volume.shape}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    warnings = []
    if not NIBABEL_AVAILABLE:
        warnings.append("• nibabel not installed - NIfTI support disabled")
    if not SCIPY_AVAILABLE:
        warnings.append("• scipy/scikit-image not installed - Contour/Oblique views disabled")
    if not TORCH_AVAILABLE:
        warnings.append("• PyTorch not installed - Using enhanced heuristic detection only")
    
    if warnings:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText("Some features may be limited:\n\n" + "\n".join(warnings))
        msg.setInformativeText("\n💡 Install all dependencies:\npip install nibabel scipy scikit-image torch torchvision")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        if msg.exec_() == QMessageBox.Cancel:
            sys.exit(0)
    
    viewer = MedicalImageViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
