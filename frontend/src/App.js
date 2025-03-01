import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import CSS file

const ImageUpload = () => {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [resultImage, setResultImage] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file)); // Show image preview before upload
    };

    const handleUpload = async () => {
        if (!image) return alert("Please select an image");

        const formData = new FormData();
        formData.append("image", image);

        setLoading(true);
        try {
            const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setResultImage(`http://127.0.0.1:5000${response.data.result_url}`);
        } catch (error) {
            alert("Error processing image");
        }
        setLoading(false);
    };

    return (
        <div className="container">
            <div className="card">
                <h2>COVID Detection from X-ray</h2>

                {/* Image Preview */}
                {preview && (
                    <div className="image-preview">
                        <p>Selected Image:</p>
                        <img src={preview} alt="Preview" className="preview-img" />
                    </div>
                )}

                {/* File Input */}
                <input type="file" onChange={handleFileChange} accept="image/png, image/jpeg" className="file-input" />

                {/* Upload Button */}
                <button onClick={handleUpload} disabled={loading} className={`upload-btn ${loading ? "disabled" : ""}`}>
                    {loading ? "Processing..." : "Upload & Analyze"}
                </button>

                {/* Result Display */}
                {resultImage && (
                    <div className="result">
                        <h3>Detection Result:</h3>
                        <img src={resultImage} alt="Detection Result" className="result-img" />
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
