import { DataTypes } from 'sequelize';
import sequelize from './index.js'; // Bạn cần tạo file config database

const Video = sequelize.define('Video', {
    id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true
    },
    userId: {
        type: DataTypes.INTEGER,
        allowNull: true,
        references: {
            model: 'users',
            key: 'id'
        }
    },
    originalName: {
        type: DataTypes.STRING,
        allowNull: false
    },
    fileName: {
        type: DataTypes.STRING,
        allowNull: false,
        unique: true
    },
    filePath: {
        type: DataTypes.STRING,
        allowNull: false
    },
    fileSize: {
        type: DataTypes.BIGINT,
        allowNull: false
    },
    mimeType: {
        type: DataTypes.STRING,
        allowNull: false
    },
    duration: {
        type: DataTypes.FLOAT,
        allowNull: true
    },
    status: {
        type: DataTypes.ENUM('uploading', 'processing', 'completed', 'failed'),
        defaultValue: 'uploading'
    },
    progress: {
        type: DataTypes.FLOAT,
        defaultValue: 0.0
    },
    isHarmful: {
        type: DataTypes.BOOLEAN,
        allowNull: true,
        defaultValue: false
    },
    category: {
        type: DataTypes.STRING,
        allowNull: true,
        comment: 'AI classification: safe, violence, sexual_content, etc.'
    },
    confidence: {
        type: DataTypes.FLOAT,
        allowNull: true,
        comment: 'AI confidence score 0-1'
    },
    llmCategory: {
        type: DataTypes.STRING,
        allowNull: true,
        comment: 'LLM classification category'
    },
    llmConfidence: {
        type: DataTypes.FLOAT,
        allowNull: true,
        comment: 'LLM confidence score 0-1'
    },
    llmExplanation: {
        type: DataTypes.TEXT,
        allowNull: true,
        comment: 'LLM explanation for classification'
    },
    llmIsHarmful: {
        type: DataTypes.BOOLEAN,
        allowNull: true,
        comment: 'LLM harmful content detection'
    },

    // Text & Vision Analysis
    transcription: {
        type: DataTypes.TEXT,
        allowNull: true,
        comment: 'Audio transcription from video'
    },
    visionAnalysis: {
        type: DataTypes.TEXT,
        allowNull: true,
        comment: 'Vision LLM analysis of video frames'
    },
    analysisResult: {
        type: DataTypes.JSONB,
        allowNull: true
    },
    summary: {
        type: DataTypes.TEXT,
        allowNull: true
    },
    errorMessage: {
        type: DataTypes.TEXT,
        allowNull: true
    }
}, {
    tableName: 'videos',
    timestamps: true,
    indexes: [
        {
            fields: ['userId']
        },
        {
            fields: ['status']
        },
        {
            fields: ['createdAt']
        }
    ]
});

// Associations
Video.associate = (models) => {
    Video.belongsTo(models.User, {
        foreignKey: 'userId',
        as: 'user'
    });
};

export default Video;