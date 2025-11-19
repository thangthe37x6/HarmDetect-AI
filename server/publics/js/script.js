// Global variables
let currentPage = 'home';
let currentVideoId = null;
let currentVideoFile = null; // THÊM MỚI
let isLoggedIn = false;
let currentUser = null;
let authMode = 'login';

// Initialize app
document.addEventListener('DOMContentLoaded', function () {
    showPage('home');
    checkAuthStatus();
    loadTheme();
    setupEventListeners();
});

function setupEventListeners() {
    // Drag and drop
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('drop', handleDrop);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('dragenter', handleDragEnter);
    }

    // Click outside to close modals
    document.addEventListener('click', function (event) {
        // Close modals when clicking outside
        if (event.target.classList.contains('modal')) {
            event.target.classList.remove('show');
        }

        // Close user dropdown
        const userDropdown = document.getElementById('user-dropdown');
        if (!event.target.closest('#user-menu') && userDropdown && !userDropdown.classList.contains('hidden')) {
            hideUserDropdown();
        }
    });
}

// Page navigation
function showPage(page) {
    // ✅ LƯU currentPage CŨ trước khi thay đổi
    const previousPage = currentPage;

    // Ẩn tất cả pages
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));

    // Hiển thị page mới
    const pageElement = document.getElementById(page + '-page');
    if (pageElement) {
        pageElement.classList.remove('hidden');
    }
    if (page === 'analyze') {
            document.getElementById('upload-section').style.display = 'block';
        }
    // ✅ KIỂM TRA: Nếu RỜI TRANG ANALYZE mà chưa lưu → Xóa video
    if (previousPage === 'analyze' && page !== 'analyze') {
        if (currentVideoId && !isVideoSaved(currentVideoId)) {
            deleteVideoFromServer(currentVideoId);
        }
    }

    // Cập nhật currentPage MỚI
    currentPage = page;

    // Load data cho pages khác
    if (page === 'history') {
        loadHistoryData();
    } else if (page === 'profile') {
        loadProfileData();
    }
}

// ==================== CLEANUP KHI ĐÓNG BROWSER - ĐÚNG ====================

window.addEventListener('beforeunload', function (e) {
    // ✅ Nếu đang ở trang analyze và chưa lưu → Xóa video
    if (currentPage === 'analyze' && currentVideoId && !isVideoSaved(currentVideoId)) {
        // Gửi request xóa (sendBeacon không bị cancel khi đóng tab)
        const url = `http://localhost:3000/api/video/${currentVideoId}`;

        // Cách 1: Dùng DELETE route
        const formData = new FormData();
        formData.append('_method', 'DELETE');
        navigator.sendBeacon(url, formData);

        // Hoặc Cách 2: Tạo endpoint riêng cho cleanup
        // navigator.sendBeacon('http://localhost:3000/api/cleanup/' + currentVideoId);
    }
});
// Mock API function -> Real API calls
async function mockAPI(endpoint, options = {}) {
    const url = `http://localhost:3000/api${endpoint}`;

    try {
        let fetchOptions = {
            method: options.method || 'GET',
            headers: {}
        };

        if (options.body && !(options.body instanceof FormData)) {
            fetchOptions.headers['Content-Type'] = 'application/json';
            fetchOptions.body = JSON.stringify(options.body);
        }

        if (options.body instanceof FormData) {
            fetchOptions.body = options.body;
        }

        const response = await fetch(url, fetchOptions);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'API call failed');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ==================== FILE HANDLING - CẬP NHẬT ====================

function handleDragEnter(e) {
    e.preventDefault();
    e.target.closest('.border-dashed').classList.add('border-gray-400', 'dark:border-gray-500');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.target.closest('.border-dashed').classList.remove('border-gray-400', 'dark:border-gray-500');
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

function handleDrop(e) {
    e.preventDefault();
    e.target.closest('.border-dashed').classList.remove('border-gray-400', 'dark:border-gray-500');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        currentVideoFile = files[0];
        processFile(files[0]);
    }
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate
    if (!file.type.startsWith('video/')) {
        showToast('Vui lòng chọn file video!');
        return;
    }
    if (file.size > 100 * 1024 * 1024) {
        showToast('File quá lớn! Vui lòng chọn file dưới 100MB');
        return;
    }
    
    currentVideoFile = file;
    
    // Hiển thị file info
    document.getElementById('file-info').classList.remove('hidden');
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    
    // Enable nút Analyze
    document.getElementById('analyze-btn').disabled = false;
    
    // ✅ GỌI HÀM CŨ - GIỮ NGUYÊN TÊN
    displayVideo(file);
}
function clearFile() {
    currentVideoFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('analyze-btn').disabled = true;
    hideVideo();
}
// Updated file processing with video display
async function processFile(file) {
    if (!file.type.startsWith('video/')) {
        showToast('Vui lòng chọn file video!');
        return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB
        showToast('File quá lớn! Vui lòng chọn file dưới 100MB');
        return;
    }

    showLoading(true);

    try {
        // HIỂN THỊ VIDEO NGAY LẬP TỨC
        displayVideo(file);

        // Create FormData for file upload
        const formData = new FormData();
        formData.append('video', file);
        if (isLoggedIn && currentUser) {
            formData.append('userId', currentUser.id);
        }

        const result = await mockAPI('/upload', {
            method: 'POST',
            body: formData
        });

        if (result.success) {
            currentVideoId = result.id;
            startAnalysis();
            showToast('Upload thành công! Bắt đầu phân tích...');
        } else {
            throw new Error(result.message || 'Upload failed');
        }
    } catch (error) {
        showToast('Upload thất bại: ' + error.message);
        hideVideo();
    } finally {
        showLoading(false);
    }
}

// ==================== VIDEO DISPLAY FUNCTIONS - MỚI ====================

function displayVideo(file) {
    const videoSection = document.getElementById('video-section');
    const videoPlayer = document.getElementById('video-player');
    const videoSource = document.getElementById('video-source');

    if (!videoSection || !videoPlayer || !videoSource) {
        console.error('Video elements not found');
        return;
    }

    // Tạo URL từ file
    const videoURL = URL.createObjectURL(file);
    videoSource.src = videoURL;
    videoPlayer.load();

    // Hiển thị video section
    videoSection.classList.remove('hidden');

    // Ẩn upload section
    // document.getElementById('upload-section').style.display = 'none';
}

function hideVideo() {
    const videoSection = document.getElementById('video-section');
    const videoPlayer = document.getElementById('video-player');
    const videoSource = document.getElementById('video-source');

    if (videoSection) videoSection.classList.add('hidden');
    if (videoSource) videoSource.src = '';
    if (videoPlayer) videoPlayer.load();
}

// ==================== ANALYSIS FUNCTIONS - CẬP NHẬT ====================
async function startAnalysisProcess() {
    if (!currentVideoFile) {
        showToast('Vui lòng chọn video!');
        return;
    }

    showLoading(true);

    try {
        // Tạo FormData
        const formData = new FormData();
        formData.append('video', currentVideoFile);
        
        // Thêm config LLM
        const useLLM = document.getElementById('use-llm').checked;
        formData.append('useLLM', useLLM ? 'true' : 'false');
        
        if (isLoggedIn && currentUser) {
            formData.append('userId', currentUser.id);
        }

        // Upload lên server
        const result = await mockAPI('/upload', {
            method: 'POST',
            body: formData
        });

        if (result.success) {
            currentVideoId = result.id;
            
            // ✅ ẨN upload section & config panel KHI BẮT ĐẦU PHÂN TÍCH
            document.getElementById('upload-section').style.display = 'none';
            
            // Bắt đầu analysis
            startAnalysis();
            showToast('Bắt đầu phân tích...');
        } else {
            throw new Error(result.message || 'Upload failed');
        }
    } catch (error) {
        showToast('Upload thất bại: ' + error.message);
        // ✅ KHÔNG ẨN upload section nếu upload thất bại
    } finally {
        showLoading(false);
    }
}
async function startAnalysis() {
    if (!currentVideoId) return;

    // Hiển thị progress section
    document.getElementById('progress-section').classList.remove('hidden');

    // Real API polling
    const pollInterval = setInterval(async () => {
        try {
            const statusResult = await mockAPI(`/status/${currentVideoId}`);

            if (statusResult.success) {
                updateProgress(statusResult.progress);

                if (statusResult.status === 'completed' || statusResult.progress >= 100) {
                    clearInterval(pollInterval);
                    await loadAnalysisResults();
                } else if (statusResult.status === 'failed') {
                    clearInterval(pollInterval);
                    showToast('Phân tích thất bại: ' + (statusResult.errorMessage || 'Unknown error'));
                    resetAnalysisUI();
                }
            } else {
                throw new Error(statusResult.message || 'Status check failed');
            }
        } catch (error) {
            clearInterval(pollInterval);
            showToast('Có lỗi trong quá trình phân tích: ' + error.message);
            resetAnalysisUI();
        }
    }, 2000); // Poll every 2 seconds
}

async function loadAnalysisResults() {
    try {
        const result = await mockAPI(`/result/${currentVideoId}`);

        if (result.success) {
            // Store results globally for saving to history
            window.currentAnalysisResult = result;
            displayResults(result); // GỌI HÀM HIỂN THỊ KẾT QUẢ
            showToast('Phân tích hoàn thành!');
        } else {
            throw new Error(result.message || 'Failed to load results');
        }
    } catch (error) {
        showToast('Không thể tải kết quả: ' + error.message);
        resetAnalysisUI();
    }
}

// ==================== DISPLAY RESULTS - MỚI ====================
function displayResults(result) {
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('results-section').classList.remove('hidden');

    const dashboardContent = document.getElementById('dashboard-content');
    if (!dashboardContent) return;

    // Phân tích dữ liệu
    const hasCNN = result.analysisResult?.details?.cnn_result;
    const hasViolence = hasCNN?.violence;
    const hasNSFW = hasCNN?.nsfw;
    const hasLLM = result.analysisResult?.details?.llm_result;
    const hasTranscription = result.analysisResult?.transcription;
    const hasVision = result.analysisResult?.vision_analysis;

    let html = '';

    // === OVERVIEW CARDS ===
    html += `
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <!-- Overall Status -->
        <div class="bg-gradient-to-br from-${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'red' : 'green'}-50 to-${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'red' : 'green'}-100 dark:from-${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'red' : 'green'}-900/20 dark:to-${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'red' : 'green'}-900/40 p-6 rounded-xl border-2 border-${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'red' : 'green'}-500">
            <div class="text-3xl mb-2">${hasViolence?.is_violent || hasNSFW?.is_nsfw ? '⚠️' : '✅'}</div>
            <p class="text-sm opacity-70 mb-1">Trạng thái tổng thể</p>
            <p class="text-2xl font-bold">${hasViolence?.is_violent || hasNSFW?.is_nsfw ? 'CÓ VẤN ĐỀ' : 'AN TOÀN'}</p>
        </div>

        <!-- Violence Score -->
        ${hasViolence ? `
        <div class="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
            <div class="flex items-center justify-between mb-2">
                <span class="text-2xl">👊</span>
                <span class="text-xs font-semibold px-2 py-1 rounded ${hasViolence.is_violent ? 'bg-red-500' : 'bg-green-500'} text-white">${hasViolence.is_violent ? 'CÓ' : 'KHÔNG'}</span>
            </div>
            <p class="text-sm opacity-70 mb-1">Bạo lực</p>
            <p class="text-2xl font-bold">${(hasViolence.violent_ratio * 100).toFixed(1)}%</p>
            <p class="text-xs mt-1 opacity-60">${(hasViolence.confidence * 100).toFixed(1)}% tin cậy</p>
        </div>
        ` : ''}

        <!-- NSFW Score -->
        ${hasNSFW ? `
        <div class="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
            <div class="flex items-center justify-between mb-2">
                <span class="text-2xl">🔞</span>
                <span class="text-xs font-semibold px-2 py-1 rounded ${hasNSFW.is_nsfw ? 'bg-red-500' : 'bg-green-500'} text-white">${hasNSFW.is_nsfw ? 'CÓ' : 'KHÔNG'}</span>
            </div>
            <p class="text-sm opacity-70 mb-1">Nhạy cảm</p>
            <p class="text-2xl font-bold">${(hasNSFW.nsfw_ratio * 100).toFixed(1)}%</p>
            <p class="text-xs mt-1 opacity-60">${(hasNSFW.confidence * 100).toFixed(1)}% tin cậy</p>
        </div>
        ` : ''}

        <!-- LLM Result -->
        ${hasLLM ? `
        <div class="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/40 p-6 rounded-xl border-2 border-purple-500">
            <div class="text-3xl mb-2">🤖</div>
            <p class="text-sm opacity-70 mb-1">AI Analysis</p>
            <p class="text-xl font-bold">${hasLLM.category}</p>
            <p class="text-xs mt-1 opacity-60">${(hasLLM.confidence * 100).toFixed(1)}% confidence</p>
        </div>
        ` : ''}
    </div>`;

    // === DETAILED ANALYSIS ===
    html += `<div class="grid md:grid-cols-2 gap-6 mb-6">`;

    // Violence Details
    if (hasViolence) {
        html += `
        <div class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
            <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
                <span>👊</span> Chi tiết bạo lực
            </h3>
            <div class="space-y-3">
                <div>
                    <div class="flex justify-between text-sm mb-1">
                        <span>Mức độ bạo lực</span>
                        <span class="font-semibold">${(hasViolence.violent_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div class="bg-red-500 h-2 rounded-full transition-all" style="width: ${(hasViolence.violent_ratio * 100)}%"></div>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-3 pt-3">
                    <div class="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p class="text-xs opacity-70">Phân loại</p>
                        <p class="font-semibold">${hasViolence.label}</p>
                    </div>
                    <div class="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p class="text-xs opacity-70">Độ tin cậy</p>
                        <p class="font-semibold">${(hasViolence.confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </div>`;
    }

    // NSFW Details
    if (hasNSFW) {
        html += `
        <div class="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm">
            <h3 class="text-lg font-bold mb-4 flex items-center gap-2">
                <span>🔞</span> Chi tiết nội dung nhạy cảm
            </h3>
            <div class="space-y-3">
                <div>
                    <div class="flex justify-between text-sm mb-1">
                        <span>Mức độ nhạy cảm</span>
                        <span class="font-semibold">${(hasNSFW.nsfw_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div class="bg-orange-500 h-2 rounded-full transition-all" style="width: ${(hasNSFW.nsfw_ratio * 100)}%"></div>
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-3 pt-3">
                    <div class="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p class="text-xs opacity-70">Phân loại</p>
                        <p class="font-semibold">${hasNSFW.label}</p>
                    </div>
                    <div class="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p class="text-xs opacity-70">Độ tin cậy</p>
                        <p class="font-semibold">${(hasNSFW.confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </div>`;
    }

    html += `</div>`;

    // === LLM EXPLANATION ===
    if (hasLLM && hasLLM.explanation) {
        html += `
        <div class="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl border-2 border-purple-400 p-6 mb-6">
            <h3 class="text-lg font-bold mb-3 flex items-center gap-2">
                <span>🤖</span> Giải thích AI
            </h3>
            <p class="text-sm leading-relaxed">${hasLLM.explanation}</p>
        </div>`;
    }

    // === ADDITIONAL INFO ===
    if (hasTranscription || hasVision) {
        html += `<div class="grid md:grid-cols-2 gap-4">`;
        
        if (hasTranscription) {
            html += `
            <details class="bg-white dark:bg-gray-800 rounded-lg border border-gray-300 dark:border-gray-600">
                <summary class="cursor-pointer px-4 py-3 font-medium hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2">
                    <span>🎤</span> Phiên âm âm thanh
                </summary>
                <div class="px-4 py-3 border-t border-gray-200 dark:border-gray-600 text-sm max-h-40 overflow-y-auto">
                    ${result.analysisResult.transcription}
                </div>
            </details>`;
        }

        if (hasVision) {
            html += `
            <details class="bg-white dark:bg-gray-800 rounded-lg border border-gray-300 dark:border-gray-600">
                <summary class="cursor-pointer px-4 py-3 font-medium hover:bg-gray-50 dark:hover:bg-gray-700 flex items-center gap-2">
                    <span>👁️</span> Phân tích hình ảnh
                </summary>
                <div class="px-4 py-3 border-t border-gray-200 dark:border-gray-600 text-sm max-h-40 overflow-y-auto">
                    ${result.analysisResult.vision_analysis}
                </div>
            </details>`;
        }

        html += `</div>`;
    }

    // === EMPTY STATE ===
    if (!hasCNN && !hasLLM && !hasTranscription && !hasVision) {
        html = `
        <div class="text-center py-16">
            <div class="text-6xl mb-4">📊</div>
            <h3 class="text-2xl font-bold mb-2">Không có kết quả</h3>
            <p class="text-gray-600 dark:text-gray-400">Vui lòng thử lại với video khác</p>
        </div>`;
    }

    dashboardContent.innerHTML = html;
}
// ===============================================================

function updateProgress(progress) {
    const progressBar = document.getElementById('progress-bar');
    const progressPercent = document.getElementById('progress-percent');
    const progressText = document.getElementById('progress-text');
    
    const clampedProgress = Math.min(Math.max(progress, 0), 100);

    progressBar.style.width = clampedProgress + '%';
    progressPercent.textContent = Math.floor(clampedProgress) + '%';
    
    // Dynamic text
    if (clampedProgress < 30) {
        progressText.textContent = 'Đang trích xuất frames...';
    } else if (clampedProgress < 60) {
        progressText.textContent = 'Đang phân tích CNN...';
    } else if (clampedProgress < 90) {
        progressText.textContent = 'Đang xử lý LLM...';
    } else {
        progressText.textContent = 'Hoàn tất!';
    }
}
function resetAnalysisUI() {
    document.getElementById('upload-section').style.display = 'block';
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    hideVideo();
    currentVideoId = null;
    currentVideoFile = null;

    // Reset file input
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';
}

// ==================== RESET ANALYSIS - MỚI ====================

async function resetAnalysis() {
    if (!currentVideoId) {
        resetAnalysisUI();
        return;
    }

    // ✅ KIỂM TRA: Nếu chưa lưu thì hỏi user
    if (!isVideoSaved(currentVideoId)) {
        const confirmDelete = confirm(
            'Video này chưa được lưu vào lịch sử!\n' +
            'Bạn có muốn phân tích video khác không?\n\n' +
            'Dữ liệu sẽ bị xóa vĩnh viễn.'
        );

        if (!confirmDelete) {
            return; // User không muốn xóa
        }

        // ✅ XÓA VIDEO khỏi PostgreSQL
        await deleteVideoFromServer(currentVideoId);
    }

    resetAnalysisUI();
    showToast('Đã reset! Bạn có thể upload video mới.');
}
// ==================== DELETE VIDEO FROM SERVER - MỚI ====================

async function deleteVideoFromServer(videoId) {
    try {
        showLoading(true);

        const response = await mockAPI(`/video/${videoId}`, {
            method: 'DELETE'
        });

        if (response.success) {
            console.log(` Video ${videoId} deleted from server`);
        } else {
            console.warn(` Failed to delete video ${videoId}:`, response.message);
        }
    } catch (error) {
        console.error('Delete video error:', error);
        // Không hiển thị lỗi cho user, vì đây là cleanup
    } finally {
        showLoading(false);
    }
}
// ==================== SAVE TO HISTORY - CẬP NHẬT ====================
async function saveToHistory() {
    if (!isLoggedIn) {
        showToast('Vui lòng đăng nhập để lưu lịch sử!');
        showAuthModal('login');
        return;
    }

    if (!window.currentAnalysisResult || !currentVideoId) {
        showToast('Chưa có kết quả để lưu!');
        return;
    }

    try {
        showLoading(true);

        // Lưu vào localStorage
        const historyItem = {
            id: Date.now(),
            videoId: currentVideoId,
            name: currentVideoFile ? currentVideoFile.name : (window.currentAnalysisResult.originalName || `Video ${currentVideoId}`),
            date: new Date().toLocaleDateString('vi-VN'),
            result: window.currentAnalysisResult.classification
                ? `${window.currentAnalysisResult.classification.is_harmful ? ' Có hại' : ' An toàn'} (${window.currentAnalysisResult.classification.category})`
                : (window.currentAnalysisResult.summary || 'Đã phân tích'),
            isHarmful: window.currentAnalysisResult.classification?.is_harmful || false,
            category: window.currentAnalysisResult.classification?.category || 'unknown',
            confidence: window.currentAnalysisResult.classification?.confidence || 0,
            data: window.currentAnalysisResult
        };

        const history = JSON.parse(localStorage.getItem('video_history') || '[]');
        history.unshift(historyItem);

        if (history.length > 50) {
            history.splice(50);
        }

        localStorage.setItem('video_history', JSON.stringify(history));

        // ✅ ĐÁNH DẤU video này đã được lưu (không xóa)
        markVideoAsSaved(currentVideoId);

        showToast('Đã lưu vào lịch sử!');
    } catch (error) {
        showToast('Lưu thất bại: ' + error.message);
    } finally {
        showLoading(false);
    }
}
function markVideoAsSaved(videoId) {
    // Lưu danh sách video đã được user lưu
    const savedVideos = JSON.parse(localStorage.getItem('saved_videos') || '[]');
    if (!savedVideos.includes(videoId)) {
        savedVideos.push(videoId);
        localStorage.setItem('saved_videos', JSON.stringify(savedVideos));
    }
}

function isVideoSaved(videoId) {
    const savedVideos = JSON.parse(localStorage.getItem('saved_videos') || '[]');
    return savedVideos.includes(videoId);
}
// ==================== AUTHENTICATION FUNCTIONS ====================

function showAuthModal(mode) {
    authMode = mode;
    const modal = document.getElementById('auth-modal');
    const title = document.getElementById('auth-title');
    const submitBtn = document.getElementById('auth-submit');
    const switchText = document.getElementById('auth-switch-text');
    const switchBtn = document.getElementById('auth-switch-btn');
    const confirmField = document.getElementById('confirm-password-field');

    if (mode === 'login') {
        title.textContent = 'Đăng nhập';
        submitBtn.textContent = 'Đăng nhập';
        switchText.textContent = 'Chưa có tài khoản?';
        switchBtn.textContent = 'Đăng ký ngay';
        confirmField.classList.add('hidden');
    } else {
        title.textContent = 'Đăng ký';
        submitBtn.textContent = 'Đăng ký';
        switchText.textContent = 'Đã có tài khoản?';
        switchBtn.textContent = 'Đăng nhập';
        confirmField.classList.remove('hidden');
    }

    modal.classList.add('show');
}

function hideAuthModal() {
    document.getElementById('auth-modal').classList.remove('show');
}

function switchAuthMode() {
    showAuthModal(authMode === 'login' ? 'register' : 'login');
}

async function handleAuth(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    if (authMode === 'register') {
        const confirmPassword = document.getElementById('confirm-password').value;
        if (password !== confirmPassword) {
            showToast('Mật khẩu không khớp!');
            return;
        }
    }

    showLoading(true);

    try {
        const result = await mockAPI('/' + authMode, {
            method: 'POST',
            body: { email, password }
        });

        if (result.success) {
            currentUser = result.user;
            isLoggedIn = true;
            saveUserSession(result.user);
            updateAuthUI();
            hideAuthModal();
            showToast(result.message || (authMode === 'login' ? 'Đăng nhập thành công!' : 'Đăng ký thành công!'));
        } else {
            throw new Error(result.message || 'Authentication failed');
        }
    } catch (error) {
        showToast('Lỗi: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function logout() {
    isLoggedIn = false;
    currentUser = null;
    localStorage.removeItem('user_session');
    updateAuthUI();
    hideUserDropdown();
    showToast('Đăng xuất thành công!');
    showPage('home');
}

function saveUserSession(user) {
    localStorage.setItem('user_session', JSON.stringify(user));
}

function checkAuthStatus() {
    const session = localStorage.getItem('user_session');
    if (session) {
        currentUser = JSON.parse(session);
        isLoggedIn = true;
        updateAuthUI();
    }
}

function updateAuthUI() {
    const authButtons = document.getElementById('auth-buttons');
    const userMenu = document.getElementById('user-menu');
    const historyNav = document.getElementById('history-nav');

    if (isLoggedIn) {
        authButtons.style.display = 'none';
        userMenu.style.display = 'block';
        historyNav.style.display = 'block';
        document.getElementById('user-name').textContent = currentUser.name || currentUser.email;
    } else {
        authButtons.style.display = 'block';
        userMenu.style.display = 'none';
        historyNav.style.display = 'none';
    }
}

function toggleUserMenu() {
    const dropdown = document.getElementById('user-dropdown');
    dropdown.classList.toggle('hidden');
}

function hideUserDropdown() {
    document.getElementById('user-dropdown').classList.add('hidden');
}

// ==================== HISTORY MANAGEMENT ====================

function loadHistoryData() {
    if (!isLoggedIn) {
        showToast('Vui lòng đăng nhập để xem lịch sử!');
        showPage('home');
        return;
    }

    const history = JSON.parse(localStorage.getItem('video_history') || '[]');
    const tableBody = document.getElementById('history-table');

    tableBody.innerHTML = '';

    if (history.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="border border-gray-300 dark:border-gray-600 p-8 text-center text-gray-500">
                    Chưa có lịch sử phân tích nào
                </td>
            </tr>
        `;
    } else {
        history.forEach(item => {
            const row = document.createElement('tr');
            const resultText = item.isHarmful
                ? `Có hại - ${item.category}`
                : `${item.result || 'An toàn'}`;

            row.innerHTML = `
            <td class="border border-gray-300 dark:border-gray-600 p-3">${item.name}</td>
            <td class="border border-gray-300 dark:border-gray-600 p-3">${item.date}</td>
            <td class="border border-gray-300 dark:border-gray-600 p-3">${resultText}</td>
                <td class="border border-gray-300 dark:border-gray-600 p-3">
                    <button onclick="deleteHistoryItem(${item.id})" class="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">Xóa</button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }
}

function deleteHistoryItem(id) {
    if (confirm('Bạn có chắc muốn xóa item này?')) {
        const history = JSON.parse(localStorage.getItem('video_history') || '[]');
        const filteredHistory = history.filter(h => h.id !== id);
        localStorage.setItem('video_history', JSON.stringify(filteredHistory));
        loadHistoryData();
        showToast('Đã xóa khỏi lịch sử!');
    }
}

// ==================== PROFILE MANAGEMENT ====================

function loadProfileData() {
    if (!isLoggedIn) {
        showPage('home');
        showToast('Vui lòng đăng nhập để xem profile!');
        return;
    }

    document.getElementById('profile-email').textContent = currentUser.email;
    document.getElementById('profile-date').textContent = new Date().toLocaleDateString('vi-VN');

    const history = JSON.parse(localStorage.getItem('video_history') || '[]');
    document.getElementById('profile-videos').textContent = history.length;
}

// ==================== THEME FUNCTIONS ====================

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    const themeToggle = document.getElementById('theme-toggle');

    if (isDark) {
        themeToggle.textContent = '☀️';
    } else {
        themeToggle.textContent = '🌙';
    }

    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = savedTheme === 'dark' || (!savedTheme && prefersDark);

    if (isDark) {
        document.documentElement.classList.add('dark');
        document.getElementById('theme-toggle').textContent = '☀️';
    }
}

// ==================== UTILITY FUNCTIONS ====================

function showLoading(show) {
    const loading = document.getElementById('loading');
    loading.style.display = show ? 'flex' : 'none';
}

function showToast(message) {
    const toast = document.getElementById('toast');
    const messageEl = document.getElementById('toast-message');

    messageEl.textContent = message;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}