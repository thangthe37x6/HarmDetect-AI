import { spawn } from 'child_process';
import fs from 'fs'
import Video from '../models/videomodel.js'
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
dotenv.config();
const USE_AI_ANALYSIS = true;
let USE_LLM = false;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || null

// ✅ QUAN TRỌNG: Đường dẫn Python trong virtual environment
const PYTHON_VENV_PATH = 'D:\\videoguard\\process_video\\Scripts\\python.exe'; // Windows
// const PYTHON_VENV_PATH = '/path/to/process_video/bin/python'; // Linux/Mac

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uploadsDir = path.join(__dirname, 'uploads')
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true })
}

// ✅ Hàm lấy Python path
function getPythonPath() {
    // Kiểm tra venv path có tồn tại không
    if (fs.existsSync(PYTHON_VENV_PATH)) {
        console.log(`✅ Using Python from venv: ${PYTHON_VENV_PATH}`);
        return PYTHON_VENV_PATH;
    }

    // Fallback về system Python
    console.warn('  Venv Python not found, using system Python');
    return process.platform === 'win32' ? 'python' : 'python3';
}
export const uploadvideo = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: 'Không có file được upload!' });
        }
        
        const { userId, useLLM } = req.body; 
        if (useLLM) {
            USE_LLM = true
        }
        const video = await Video.create({
            userId: userId || null,
            originalName: req.file.originalname,
            fileName: req.file.filename,
            filePath: req.file.path,
            fileSize: req.file.size,
            mimeType: req.file.mimetype,
            status: 'processing',
            progress: 0,
        });

        // TRUYỀN useLLM VÀO processVideoAsync
        processVideoAsync(video.id, req.file.path, useLLM === 'true');

        res.json({
            success: true,
            message: 'Video uploaded successfully',
            id: video.id
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({
            success: false,
            message: error.message || 'Upload failed'
        });
    }
}

export const checkstatus = async (req, res) => {
    try {
        const { VideoId } = req.params;

        const video = await Video.findByPk(VideoId);
        if (!video) {
            return res.status(404).json({
                success: false,
                message: 'Video not found'
            });
        }

        res.json({
            success: true,
            status: video.status,
            progress: video.progress,
            errorMessage: video.errorMessage
        });
    } catch (error) {
        console.error('Status check error:', error);
        res.status(500).json({
            success: false,
            message: error.message || 'Status check failed'
        });
    }
};

export const getresult = async (req, res) => {
    try {
        const { VideoId } = req.params;
        const video = await Video.findByPk(VideoId);

        if (!video) {
            return res.status(404).json({
                success: false,
                message: 'Video not found'
            });
        }

        if (video.status !== 'completed') {
            return res.status(400).json({
                success: false,
                message: 'Video analysis not completed yet',
                status: video.status,
                progress: video.progress
            });
        }

        res.json({
            success: true,
            originalName: video.originalName,
            summary: video.summary,
            analysisResult: video.analysisResult,
            detections: video.detections,
            threshold: video.threshold,
            duration: video.duration,
            createdAt: video.createdAt
        });
    } catch (error) {
        console.error('Result fetch error:', error);
        res.status(500).json({
            success: false,
            message: error.message || 'Failed to fetch result'
        });
    }
}

async function processVideoAsync(VideoId, videopath, useLLM = false) {
    let progressInterval = null;

    try {
        await Video.update({ progress: 10 }, { where: { id: VideoId } });

        const pythonPath = path.join(__dirname, "../videoProcessor.py");
        const args = [pythonPath, videopath];

        // SỬ DỤNG THAM SỐ useLLM
        if (USE_AI_ANALYSIS && useLLM) {
            args.push('--use-llm');
            if (OPENAI_API_KEY) {
                args.push(`--api-key=${OPENAI_API_KEY}`);
            }
        }

        // ✅ Sử dụng Python từ venv
        const pythonExecutable = getPythonPath();
        console.log(` Starting Python process with: ${pythonExecutable}`);

        const python = spawn(pythonExecutable, args);

        let stdoutBuffer = '';
        let stderrData = '';

        python.stdout.on('data', (data) => {
            const output = data.toString();

            // Gom tất cả output lại (bao gồm cả JSON cuối)
            stdoutBuffer += output;

            // In ra log nếu không phải JSON
            if (!output.trim().startsWith('{') && !output.trim().startsWith('[')) {
                console.log(`Python: ${output.trim()}`);
            }
        });

        python.stderr.on('data', (data) => {
            const error = data.toString();
            stderrData += error;

            // ✅ LỌC CHỈ GHI LỖI THẬT, BỎ QUA WARNING/INFO
            const lowerError = error.toLowerCase();
            if (!lowerError.includes('tensorflow') &&
                !lowerError.includes('onednn') &&
                !lowerError.includes('cpu_feature_guard') &&
                !lowerError.includes('loaded violence model') &&
                !lowerError.includes('loaded nsfw model')) {
                console.error(`Python stderr: ${error.trim()}`);
            }
        });

        python.on('close', async (code) => {
            if (progressInterval) clearInterval(progressInterval);

            if (code !== 0) {
                console.error(`Python process exited with code ${code}`);
                console.error(`stderr output: ${stderrData}`);
                await Video.update({
                    status: 'failed',
                    progress: 0,
                    errorMessage: `Python process failed: ${stderrData || 'Unknown error'}`
                }, { where: { id: VideoId } });
                return;
            }

            try {
                // 🔧 FIX: Tách log text và JSON
                const lines = stdoutBuffer.trim().split('\n');

                // Tìm dòng cuối cùng là JSON hợp lệ
                let jsonString = null;
                for (let i = lines.length - 1; i >= 0; i--) {
                    const line = lines[i].trim();
                    if (line.startsWith('{') && line.endsWith('}')) {
                        try {
                            JSON.parse(line); // Test parse
                            jsonString = line;
                            break;
                        } catch (e) {
                            continue; // Không phải JSON hợp lệ, thử dòng trước
                        }
                    }
                }

                if (!jsonString) {
                    throw new Error('No valid JSON found in Python output');
                }

                const result = JSON.parse(jsonString);

                if (result.success) {
                    const updateData = {
                        status: 'completed',
                        progress: 100,
                        analysisResult: result,
                        summary: result.summary || 'Analysis completed',
                        duration: result.video_info?.duration || result.metadata?.duration || null
                    };

                    if (result.classification) {
                        updateData.isHarmful = result.classification.is_harmful;
                        updateData.category = result.classification.category;
                        updateData.confidence = result.classification.confidence;
                    }
                    if (result.details?.llm_result) {
                        updateData.llmCategory = result.details.llm_result.category;
                        updateData.llmConfidence = result.details.llm_result.confidence;
                        updateData.llmExplanation = result.details.llm_result.explanation;
                        updateData.llmIsHarmful = result.details.llm_result.is_harmful;
                    }

                    // THÊM ĐOẠN NÀY:
                    if (result.transcription) {
                        updateData.transcription = result.transcription;
                    }
                    if (result.vision_analysis) {
                        updateData.visionAnalysis = result.vision_analysis;
                    }

                    await Video.update(updateData, { where: { id: VideoId } });

                    if (result.classification?.is_harmful) {
                        console.log(`🚨 Video ${VideoId} - HARMFUL CONTENT DETECTED`);
                        console.log(`   Category: ${result.classification.category}`);
                        console.log(`   Confidence: ${(result.classification.confidence * 100).toFixed(1)}%`);
                    } else {
                        console.log(`✅ Video ${VideoId} - Safe content`);
                    }
                } else {
                    await Video.update({
                        status: 'failed',
                        progress: 0,
                        errorMessage: result.error || 'Analysis failed'
                    }, { where: { id: VideoId } });
                    console.error(`❌ Video ${VideoId} analysis failed:`, result.error);
                }

            } catch (parseError) {
                console.error('Failed to parse Python output:', parseError);
                console.error('Raw Python output:', stdoutBuffer);
                await Video.update({
                    status: 'failed',
                    progress: 0,
                    errorMessage: 'Failed to parse analysis result'
                }, { where: { id: VideoId } });
            }
        });
        progressInterval = setInterval(async () => {
            try {
                const video = await Video.findByPk(VideoId);
                if (video && video.status === 'processing' && video.progress < 90) {
                    await Video.update(
                        { progress: Math.min(video.progress + 10, 90) },
                        { where: { id: VideoId } }
                    );
                } else {
                    clearInterval(progressInterval);
                }
            } catch (err) {
                console.error('Progress update error:', err);
                clearInterval(progressInterval);
            }
        }, 2000);

    } catch (error) {
        console.error('Process video error:', error);

        if (progressInterval) {
            clearInterval(progressInterval);
        }

        await Video.update(
            {
                status: 'failed',
                progress: 0,
                errorMessage: error.message
            },
            { where: { id: VideoId } }
        );
    }
}

export const deletevideo = async (req, res) => {
    try {
        const { VideoId } = req.params;

        const video = await Video.findByPk(VideoId);
        if (!video) {
            return res.status(404).json({
                success: false,
                message: 'Video not found'
            });
        }

        if (fs.existsSync(video.filePath)) {
            fs.unlinkSync(video.filePath);
            console.log(`🗑️  Deleted file: ${video.filePath}`);
        }

        await video.destroy();

        res.json({
            success: true,
            message: 'Video deleted successfully'
        });
    } catch (error) {
        console.error('Delete video error:', error);
        res.status(500).json({
            success: false,
            message: error.message || 'Delete failed'
        });
    }
}

