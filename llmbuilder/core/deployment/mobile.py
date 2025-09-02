"""
Mobile deployment export for LLMBuilder.

This module provides functionality for preparing models for mobile
deployment on Android and iOS platforms.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import tempfile
from datetime import datetime

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class MobileExporter:
    """
    Exports models for mobile deployment.
    
    Prepares model weights and generates integration code for Android/iOS
    deployment with optimizations for mobile devices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mobile exporter.
        
        Args:
            config: Export configuration dictionary
        """
        self.config = config
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Mobile exporter initialized")
    
    def _validate_config(self):
        """Validate export configuration."""
        required_keys = ['model_path', 'output_dir', 'platform']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate paths
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Validate platform
        platform = self.config['platform']
        if platform not in ['android', 'ios', 'both']:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback for monitoring export progress."""
        self.progress_callback = callback
    
    def _report_progress(self, step: str, percentage: float):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(step, percentage)
        else:
            logger.info(f"{step} ({percentage:.1f}%)")
    
    def export(self) -> Dict[str, Any]:
        """
        Export model for mobile deployment.
        
        Returns:
            Dictionary with export results and metadata
        """
        self._report_progress("Starting mobile export", 5)
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        platform = self.config['platform']
        results = {'platforms': {}}
        
        if platform in ['android', 'both']:
            self._report_progress("Exporting for Android", 20)
            android_result = self._export_android(output_dir / 'android')
            results['platforms']['android'] = android_result
        
        if platform in ['ios', 'both']:
            self._report_progress("Exporting for iOS", 60)
            ios_result = self._export_ios(output_dir / 'ios')
            results['platforms']['ios'] = ios_result
        
        # Create common documentation
        self._report_progress("Creating documentation", 90)
        self._create_mobile_documentation(output_dir)
        
        self._report_progress("Mobile export complete", 100)
        
        results.update({
            'output_dir': str(output_dir),
            'quantization': self.config.get('quantization', 'q8_0'),
            'created_at': datetime.now().isoformat()
        })
        
        return results
    
    def _export_android(self, android_dir: Path) -> Dict[str, Any]:
        """Export for Android platform."""
        android_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create Android-specific directories
            (android_dir / 'assets').mkdir(exist_ok=True)
            (android_dir / 'java').mkdir(exist_ok=True)
            (android_dir / 'cpp').mkdir(exist_ok=True)
            
            # Copy and optimize model
            model_path = Path(self.config['model_path'])
            optimized_model = self._optimize_model_for_mobile(model_path, 'android')
            
            # Copy optimized model to assets
            shutil.copy2(optimized_model, android_dir / 'assets' / 'model.bin')
            
            # Generate Android integration code
            self._generate_android_code(android_dir)
            
            # Create Android-specific configuration
            android_config = self._create_android_config()
            with open(android_dir / 'config.json', 'w') as f:
                json.dump(android_config, f, indent=2)
            
            # Calculate model size
            model_size = (android_dir / 'assets' / 'model.bin').stat().st_size
            
            return {
                'success': True,
                'model_size': self._format_size(model_size),
                'files': list(android_dir.rglob('*')),
                'integration_files': [
                    'java/ModelInference.java',
                    'cpp/model_jni.cpp',
                    'assets/model.bin'
                ]
            }
            
        except Exception as e:
            logger.error(f"Android export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'files': []
            }
    
    def _export_ios(self, ios_dir: Path) -> Dict[str, Any]:
        """Export for iOS platform."""
        ios_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create iOS-specific directories
            (ios_dir / 'Resources').mkdir(exist_ok=True)
            (ios_dir / 'Sources').mkdir(exist_ok=True)
            (ios_dir / 'Headers').mkdir(exist_ok=True)
            
            # Copy and optimize model
            model_path = Path(self.config['model_path'])
            optimized_model = self._optimize_model_for_mobile(model_path, 'ios')
            
            # Copy optimized model to resources
            shutil.copy2(optimized_model, ios_dir / 'Resources' / 'model.bin')
            
            # Generate iOS integration code
            self._generate_ios_code(ios_dir)
            
            # Create iOS-specific configuration
            ios_config = self._create_ios_config()
            with open(ios_dir / 'config.json', 'w') as f:
                json.dump(ios_config, f, indent=2)
            
            # Calculate model size
            model_size = (ios_dir / 'Resources' / 'model.bin').stat().st_size
            
            return {
                'success': True,
                'model_size': self._format_size(model_size),
                'files': list(ios_dir.rglob('*')),
                'integration_files': [
                    'Sources/ModelInference.swift',
                    'Headers/ModelInference.h',
                    'Resources/model.bin'
                ]
            }
            
        except Exception as e:
            logger.error(f"iOS export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'files': []
            }
    
    def _optimize_model_for_mobile(self, model_path: Path, platform: str) -> Path:
        """Optimize model for mobile deployment."""
        # For now, just copy the model
        # In a real implementation, this would:
        # 1. Apply mobile-specific quantization
        # 2. Optimize for target hardware
        # 3. Convert to mobile-friendly format
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            temp_path = Path(f.name)
        
        # Simple copy for demonstration
        shutil.copy2(model_path, temp_path)
        
        # Check size constraints
        max_size = self.config.get('max_model_size', 100) * 1024 * 1024  # MB to bytes
        actual_size = temp_path.stat().st_size
        
        if actual_size > max_size:
            logger.warning(f"Model size ({self._format_size(actual_size)}) exceeds limit ({self._format_size(max_size)})")
        
        return temp_path
    
    def _generate_android_code(self, android_dir: Path):
        """Generate Android integration code."""
        # Java wrapper class
        java_code = '''package com.example.llmbuilder;

import android.content.Context;
import android.content.res.AssetManager;
import java.io.IOException;
import java.io.InputStream;

public class ModelInference {
    private static final String MODEL_FILE = "model.bin";
    private boolean isModelLoaded = false;
    
    static {
        System.loadLibrary("llmbuilder_jni");
    }
    
    public ModelInference(Context context) {
        loadModel(context);
    }
    
    private void loadModel(Context context) {
        try {
            AssetManager assetManager = context.getAssets();
            InputStream inputStream = assetManager.open(MODEL_FILE);
            
            // Load model through JNI
            isModelLoaded = loadModelNative(inputStream);
            inputStream.close();
            
        } catch (IOException e) {
            e.printStackTrace();
            isModelLoaded = false;
        }
    }
    
    public String generateText(String prompt, int maxTokens, float temperature) {
        if (!isModelLoaded) {
            return "Model not loaded";
        }
        
        return generateTextNative(prompt, maxTokens, temperature);
    }
    
    public boolean isReady() {
        return isModelLoaded;
    }
    
    // Native methods
    private native boolean loadModelNative(InputStream modelStream);
    private native String generateTextNative(String prompt, int maxTokens, float temperature);
}
'''
        
        with open(android_dir / 'java' / 'ModelInference.java', 'w') as f:
            f.write(java_code)
        
        # JNI C++ code
        cpp_code = '''#include <jni.h>
#include <string>
#include <android/log.h>

#define LOG_TAG "LLMBuilder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_llmbuilder_ModelInference_loadModelNative(JNIEnv *env, jobject thiz, jobject model_stream) {
    LOGI("Loading model from stream");
    
    // TODO: Implement actual model loading
    // This is a placeholder implementation
    
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_example_llmbuilder_ModelInference_generateTextNative(JNIEnv *env, jobject thiz, jstring prompt, jint max_tokens, jfloat temperature) {
    const char *prompt_str = env->GetStringUTFChars(prompt, 0);
    
    LOGI("Generating text for prompt: %s", prompt_str);
    
    // TODO: Implement actual text generation
    // This is a placeholder implementation
    std::string result = std::string(prompt_str) + " [Generated on Android with " + std::to_string(max_tokens) + " tokens]";
    
    env->ReleaseStringUTFChars(prompt, prompt_str);
    
    return env->NewStringUTF(result.c_str());
}

}
'''
        
        with open(android_dir / 'cpp' / 'model_jni.cpp', 'w') as f:
            f.write(cpp_code)
        
        # CMakeLists.txt for building JNI
        cmake_code = '''cmake_minimum_required(VERSION 3.4.1)

add_library(llmbuilder_jni SHARED
    model_jni.cpp
)

find_library(log-lib log)

target_link_libraries(llmbuilder_jni ${log-lib})
'''
        
        with open(android_dir / 'cpp' / 'CMakeLists.txt', 'w') as f:
            f.write(cmake_code)
    
    def _generate_ios_code(self, ios_dir: Path):
        """Generate iOS integration code."""
        # Swift wrapper class
        swift_code = '''import Foundation

@objc public class ModelInference: NSObject {
    private var isModelLoaded = false
    private let modelFileName = "model.bin"
    
    public override init() {
        super.init()
        loadModel()
    }
    
    private func loadModel() {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "bin") else {
            print("Model file not found in bundle")
            return
        }
        
        // TODO: Implement actual model loading
        // This is a placeholder implementation
        isModelLoaded = true
        print("Model loaded from: \\(modelPath)")
    }
    
    @objc public func generateText(prompt: String, maxTokens: Int, temperature: Float) -> String {
        guard isModelLoaded else {
            return "Model not loaded"
        }
        
        // TODO: Implement actual text generation
        // This is a placeholder implementation
        return "\\(prompt) [Generated on iOS with \\(maxTokens) tokens]"
    }
    
    @objc public var isReady: Bool {
        return isModelLoaded
    }
}
'''
        
        with open(ios_dir / 'Sources' / 'ModelInference.swift', 'w') as f:
            f.write(swift_code)
        
        # Objective-C header
        objc_header = '''#import <Foundation/Foundation.h>

@interface ModelInference : NSObject

- (instancetype)init;
- (NSString *)generateTextWithPrompt:(NSString *)prompt maxTokens:(NSInteger)maxTokens temperature:(float)temperature;
- (BOOL)isReady;

@end
'''
        
        with open(ios_dir / 'Headers' / 'ModelInference.h', 'w') as f:
            f.write(objc_header)
    
    def _create_android_config(self) -> Dict[str, Any]:
        """Create Android-specific configuration."""
        return {
            'platform': 'android',
            'model_file': 'model.bin',
            'quantization': self.config.get('quantization', 'q8_0'),
            'max_tokens': 512,
            'default_temperature': 0.8,
            'jni_library': 'llmbuilder_jni',
            'min_sdk_version': 21,
            'target_sdk_version': 33,
            'permissions': [
                'android.permission.INTERNET'
            ]
        }
    
    def _create_ios_config(self) -> Dict[str, Any]:
        """Create iOS-specific configuration."""
        return {
            'platform': 'ios',
            'model_file': 'model.bin',
            'quantization': self.config.get('quantization', 'q8_0'),
            'max_tokens': 512,
            'default_temperature': 0.8,
            'min_ios_version': '12.0',
            'frameworks': [
                'Foundation',
                'CoreML'
            ]
        }
    
    def _create_mobile_documentation(self, output_dir: Path):
        """Create mobile deployment documentation."""
        readme_content = f'''# LLMBuilder Mobile Deployment

This package contains optimized models and integration code for mobile deployment.

## Supported Platforms

- Android (API level 21+)
- iOS (12.0+)

## Package Contents

### Android
- `android/assets/model.bin` - Optimized model file
- `android/java/ModelInference.java` - Java wrapper class
- `android/cpp/model_jni.cpp` - JNI implementation
- `android/cpp/CMakeLists.txt` - Build configuration

### iOS
- `ios/Resources/model.bin` - Optimized model file
- `ios/Sources/ModelInference.swift` - Swift wrapper class
- `ios/Headers/ModelInference.h` - Objective-C header

## Integration Instructions

### Android

1. Copy the `android/` directory contents to your Android project
2. Add the JNI library to your `build.gradle`:
   ```gradle
   android {{
       externalNativeBuild {{
           cmake {{
               path "src/main/cpp/CMakeLists.txt"
           }}
       }}
   }}
   ```
3. Use the ModelInference class in your Java/Kotlin code:
   ```java
   ModelInference model = new ModelInference(context);
   String result = model.generateText("Hello", 50, 0.8f);
   ```

### iOS

1. Copy the `ios/` directory contents to your iOS project
2. Add the model file to your app bundle
3. Use the ModelInference class in your Swift code:
   ```swift
   let model = ModelInference()
   let result = model.generateText(prompt: "Hello", maxTokens: 50, temperature: 0.8)
   ```

## Model Information

- Quantization: {self.config.get('quantization', 'q8_0')}
- Max model size: {self.config.get('max_model_size', 100)} MB
- Created: {datetime.now().isoformat()}

## Performance Considerations

- Model loading may take several seconds on first run
- Consider loading the model asynchronously
- Monitor memory usage during inference
- Test on target devices for performance validation

## Support

For more information and updates, visit: https://github.com/your-org/llmbuilder
'''
        
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"