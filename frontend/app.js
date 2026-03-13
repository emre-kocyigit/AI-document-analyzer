const API_BASE = 'http://127.0.0.1:8000/api/v1';

// DOM Elements
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const insightsDashboard = document.getElementById('insights-dashboard');
const docNameEl = document.getElementById('doc-name');
const classificationTags = document.getElementById('classification-tags');
const entityList = document.getElementById('entity-list');

const chatBadge = document.getElementById('chat-badge');
const chatMessages = document.getElementById('chat-messages');
const chatInputArea = document.getElementById('chat-input-area');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const apiStatusDot = document.getElementById('api-status-dot');
const apiStatusText = document.getElementById('api-status-text');

let currentDocId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    setupUploadHandlers();
    setupChatHandlers();
});

// --- API Health ---
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        apiStatusDot.className = 'pulse-dot online';
        apiStatusText.textContent = data.status.includes('degraded') ? 'API Online (Ollama Offline)' : 'API Online';
        apiStatusText.style.color = data.status.includes('degraded') ? '#f59e0b' : '#10b981';
    } catch (error) {
        apiStatusDot.className = 'pulse-dot error';
        apiStatusText.textContent = 'API Offline';
        apiStatusText.style.color = '#ef4444';
        console.error('API Check Failed', error);
    }
}

// --- Upload Handlers ---
function setupUploadHandlers() {
    // Click to upload
    uploadZone.addEventListener('click', () => {
        if (!uploadProgress.classList.contains('active')) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    // UI Loading state
    uploadProgress.classList.add('active');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        
        // Success
        currentDocId = data.doc_id;
        docNameEl.textContent = data.filename;
        
        // Render Insights
        renderInsights(data.analysis);
        
        // Update UI States
        uploadZone.classList.add('hidden');
        insightsDashboard.classList.remove('hidden');
        
        unlockChatState();

    } catch (error) {
        console.error(error);
        alert('Failed to analyze document. Is the backend running?');
    } finally {
        uploadProgress.classList.remove('active');
        fileInput.value = ''; // reset
    }
}

function renderInsights(analysis) {
    // Clear existing
    classificationTags.innerHTML = '';
    entityList.innerHTML = '';

    // Classifications
    if (analysis && analysis.classifications && analysis.classifications.length > 0) {
        analysis.classifications.forEach(cls => {
            const span = document.createElement('span');
            span.className = 'tag';
            span.textContent = `${cls.label} (${Math.round(cls.confidence * 100)}%)`;
            classificationTags.appendChild(span);
        });
    } else {
        classificationTags.innerHTML = '<span class="subtitle">No categories detected</span>';
    }

    // Entities
    if (analysis && analysis.entities && analysis.entities.length > 0) {
        // Group by label if you wanted, but here we just list them
        analysis.entities.forEach(ent => {
            const div = document.createElement('div');
            div.className = 'entity';
            div.innerHTML = `<span class="entity-label">${ent.label}</span> ${ent.text}`;
            entityList.appendChild(div);
        });
    } else {
        entityList.innerHTML = '<span class="subtitle">No entities extracted</span>';
    }
}

// --- Chat Interface ---
function unlockChatState() {
    chatBadge.textContent = 'Ready';
    chatBadge.className = 'badge ready';
    
    chatInputArea.classList.remove('locked');
    chatInput.disabled = false;
    sendBtn.disabled = false;

    // Add unprompted bot message
    appendMessage('bot', `I've analyzed the document. You can now ask me questions about its contents.`);
}

function setupChatHandlers() {
    sendBtn.addEventListener('click', handleSendChat);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSendChat();
        }
    });
}

async function handleSendChat() {
    const text = chatInput.value.trim();
    if (!text || !currentDocId) return;

    // Add User Message
    appendMessage('user', text);
    chatInput.value = '';
    
    // Disable inputs while waiting
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Add loading indicator
    const typingId = appendTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/documents/${currentDocId}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_id: currentDocId,
                question: text
            })
        });

        if (!response.ok) throw new Error('Failed to fetch answer');

        const data = await response.json();
        
        removeMessage(typingId);
        appendMessage('bot', data.answer);

    } catch (error) {
        removeMessage(typingId);
        appendMessage('bot', 'Sorry, I encountered an error while trying to answer your question. Is Ollama running?');
        console.error(error);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
        scrollToBottom();
    }
}

function appendMessage(sender, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${sender === 'user' ? 'user-msg' : 'system-msg'}`;
    
    const icon = sender === 'user' ? 'user' : 'bot';
    
    wrapper.innerHTML = `
        <div class="msg-avatar"><i data-lucide="${icon}"></i></div>
        <div class="msg-bubble">${escapeHTML(text)}</div>
    `;
    
    chatMessages.appendChild(wrapper);
    lucide.createIcons({ root: wrapper });
    scrollToBottom();
}

function appendTypingIndicator() {
    const id = 'typing-' + Date.now();
    const wrapper = document.createElement('div');
    wrapper.id = id;
    wrapper.className = 'message system-msg';
    
    wrapper.innerHTML = `
        <div class="msg-avatar"><i data-lucide="bot"></i></div>
        <div class="msg-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(wrapper);
    lucide.createIcons({ root: wrapper });
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Simple security against XSS in chat
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag])
    );
}
