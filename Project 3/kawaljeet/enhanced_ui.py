"""
Enhanced ChefBot Web UI with File Upload Support
Flask-based web interface with PDF upload and Image OCR capabilities
"""
import os
import sys
import tempfile
from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Lazy load the bot
bot = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_bot():
    """Lazy initialization of ChefBot - loads Phi-2 model on first use"""
    global bot
    if bot is None:
        print("🤖 Loading ChefBot with Phi-2 model (this may take 30-60 seconds)...")
        from app import ChefBotRAG
        bot = ChefBotRAG()
        print("✅ ChefBot ready!")
    return bot

# HTML Template with File Upload
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChefBot - Cooking Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            height: 90vh;
        }
        .chat-section {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .sidebar {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 20px;
            overflow-y: auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { font-size: 13px; opacity: 0.9; }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f7f7f7;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin: 0 10px;
            flex-shrink: 0;
        }
        .message.user .avatar { background: #764ba2; order: 2; }
        .message.bot .avatar { background: #ff6b6b; order: 1; }
        .timing {
            font-size: 11px;
            color: #999;
            margin-top: 5px;
            font-style: italic;
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        .input-row {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .input-container input[type="text"] {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: border 0.3s;
        }
        .input-container input[type="text"]:focus { border-color: #667eea; }
        .btn {
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 15px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
            font-weight: bold;
        }
        .loading.active { display: block; }
        .sidebar h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .sidebar-section {
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
        }
        .sidebar-section:last-child { border-bottom: none; }
        .toggle-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #f7f7f7;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .toggle-row label { font-size: 14px; color: #666; }
        input[type="checkbox"] {
            width: 40px;
            height: 22px;
            cursor: pointer;
        }
        .file-upload {
            margin-top: 10px;
        }
        .file-upload input[type="file"] {
            display: none;
        }
        .file-upload label {
            display: block;
            padding: 10px;
            background: #f7f7f7;
            border: 2px dashed #667eea;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-upload label:hover {
            background: #e7e7ff;
            border-color: #764ba2;
        }
        .file-upload label i { font-size: 24px; display: block; margin-bottom: 5px; }
        .upload-status {
            margin-top: 10px;
            padding: 8px;
            border-radius: 8px;
            font-size: 13px;
            display: none;
        }
        .upload-status.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        .upload-status.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
        .clear-btn {
            padding: 8px 16px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
        }
        .clear-btn:hover { background: #c82333; }
        @media (max-width: 968px) {
            .main-container {
                grid-template-columns: 1fr;
                height: auto;
            }
            .sidebar { margin-top: 20px; }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Chat Section -->
        <div class="chat-section">
            <div class="header">
                <h1>👨‍🍳 ChefBot</h1>
                <p>AI Cooking Assistant</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot">
                    <div class="avatar">👨‍🍳</div>
                    <div class="message-content">
                        <p>Hi! I'm ChefBot, your cooking assistant ready to help with recipes, techniques, and cooking tips.</p>
                    </div>
                </div>
            </div>
            
            <div class="loading" id="loading">
                🤔 ChefBot is thinking... (This may take 15-30 seconds)
            </div>
            
            <div class="input-container">
                <div class="input-row">
                    <input type="text" id="userInput" placeholder="Ask me about cooking..." 
                           onkeypress="handleKeyPress(event)">
                    <button class="btn" onclick="sendMessage()" id="sendBtn">Send 🚀</button>
                    <button class="clear-btn" onclick="clearChat()">Clear</button>
                </div>
            </div>
        </div>
        
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Upload PDF -->
            <div class="sidebar-section">
                <h3>📄 Upload PDF</h3>
                <div class="file-upload">
                    <input type="file" id="pdfFile" accept=".pdf" onchange="uploadPDF()">
                    <label for="pdfFile">
                        📁
                        <div>Click to upload PDF</div>
                        <small>Add cooking knowledge</small>
                    </label>
                </div>
                <div class="upload-status" id="pdfStatus"></div>
            </div>
            
            <!-- Upload Image -->
            <div class="sidebar-section">
                <h3>🖼️ Upload Image</h3>
                <div class="file-upload">
                    <input type="file" id="imageFile" accept=".png,.jpg,.jpeg" onchange="uploadImage()">
                    <label for="imageFile">
                        📷
                        <div>Click to upload image</div>
                        <small>Extract text (OCR)</small>
                    </label>
                </div>
                <div class="upload-status" id="imageStatus"></div>
                <small style="color: #999; display: block; margin-top: 5px;">
                    Note: Requires Tesseract OCR
                </small>
            </div>
        </div>
    </div>

    <script>
        let messageHistory = [];
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            const loading = document.getElementById('loading');
            const sendBtn = document.getElementById('sendBtn');
            loading.classList.add('active');
            sendBtn.disabled = true;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                const timing = `⏱️ Total: ${data.total_time}s | Retrieval: ${data.retrieval_time}s | Generation: ${data.generation_time}s`;
                addMessage(data.response, 'bot', timing);
                
            } catch (error) {
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                console.error('Error:', error);
            } finally {
                loading.classList.remove('active');
                sendBtn.disabled = false;
            }
        }
        
        function addMessage(text, sender, timing = '') {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = sender === 'user' ? '👤' : '👨‍🍳';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text;
            
            if (timing) {
                const timingDiv = document.createElement('div');
                timingDiv.className = 'timing';
                timingDiv.textContent = timing;
                contentDiv.appendChild(timingDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            container.appendChild(messageDiv);
            
            container.scrollTop = container.scrollHeight;
            
            messageHistory.push({ text, sender, timing });
        }
        
        async function uploadPDF() {
            const fileInput = document.getElementById('pdfFile');
            const statusDiv = document.getElementById('pdfStatus');
            
            if (!fileInput.files.length) return;
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            statusDiv.textContent = '⏳ Uploading and processing PDF...';
            statusDiv.className = 'upload-status';
            statusDiv.style.display = 'block';
            
            try {
                const response = await fetch('/upload_pdf', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                statusDiv.textContent = data.message;
                statusDiv.className = data.success ? 'upload-status success' : 'upload-status error';
            } catch (error) {
                statusDiv.textContent = '❌ Upload failed: ' + error.message;
                statusDiv.className = 'upload-status error';
            }
            
            fileInput.value = '';
        }
        
        async function uploadImage() {
            const fileInput = document.getElementById('imageFile');
            const statusDiv = document.getElementById('imageStatus');
            
            if (!fileInput.files.length) return;
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            statusDiv.textContent = '⏳ Uploading and extracting text...';
            statusDiv.className = 'upload-status';
            statusDiv.style.display = 'block';
            
            try {
                const response = await fetch('/upload_image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                statusDiv.textContent = data.message;
                statusDiv.className = data.success ? 'upload-status success' : 'upload-status error';
            } catch (error) {
                statusDiv.textContent = '❌ Upload failed: ' + error.message;
                statusDiv.className = 'upload-status error';
            }
            
            fileInput.value = '';
        }
        
        async function clearChat() {
            if (!confirm('Clear chat history? (ChefBot will forget the conversation)')) return;
            
            try {
                await fetch('/clear_memory', { method: 'POST' });
                document.getElementById('chatContainer').innerHTML = '';
                messageHistory = [];
                addMessage('Chat cleared! I\\'ve forgotten our previous conversation.', 'bot');
            } catch (error) {
                console.error('Error clearing chat:', error);
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        chefbot = get_bot()
        print(f"\n📩 Processing: {message[:50]}...")
        result = chefbot.chat(message)
        print(f"✅ Response: {result['response'][:100]}...")
        print(f"⏱️  Time: {result['total_time']}s")
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error: {error_details}")
        return jsonify({
            'response': f'Sorry, I encountered an error: {str(e)}',
            'total_time': 0,
            'retrieval_time': 0,
            'generation_time': 0
        })

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type'})
        
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean up
        os.remove(filepath)
        
        if not text.strip():
            return jsonify({'success': False, 'message': 'No text found in PDF'})
        
        # Add to knowledge base
        chefbot = get_bot()
        num_chunks = chefbot.add_document(text, source=filename)
        
        return jsonify({
            'success': True,
            'message': f'✅ Added {num_chunks} chunks from {filename}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type'})
        
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        chefbot = get_bot()
        text = chefbot.extract_text_from_image(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if not text.strip():
            return jsonify({'success': False, 'message': 'No text found in image'})
        
        # Add to knowledge base
        num_chunks = chefbot.add_document(text, source=filename)
        
        return jsonify({
            'success': True,
            'message': f'✅ Added {num_chunks} chunks from image\\nExtracted: {text[:100]}...'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}\\nNote: Tesseract OCR must be installed'})

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    try:
        chefbot = get_bot()
        chefbot.reset_memory()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ChefBot Enhanced UI with File Uploads")
    print("="*60)
    print("📝 Note: Bot loads on first message (~30-60 sec)")
    print("🧠 Model: Microsoft Phi-2 (2.7B) with 8-bit quantization")
    print("💾 Memory: Remembers last 5 conversations")
    print("📤 Features: PDF upload, Image OCR, RAG toggle")
    print("🌐 URL: http://127.0.0.1:5000")
    print("⚠️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
