-- ============================================================
-- schema.sql — MySQL Database Schema
-- AI Image Emotion Recognition System
-- ============================================================
-- Run this file once to create the database and table:
--   mysql -u root -p < schema.sql
-- ============================================================

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS emotion_recognition_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Use the database for subsequent statements
USE emotion_recognition_db;

-- ── emotion_results table ────────────────────────────────────
-- Stores one row per emotion-detection prediction.
CREATE TABLE IF NOT EXISTS emotion_results (
    id                INT          NOT NULL AUTO_INCREMENT,
    image_name        VARCHAR(255) NOT NULL COMMENT 'Original uploaded filename',
    predicted_emotion VARCHAR(50)  NOT NULL COMMENT 'Top-1 emotion label (e.g. happy)',
    confidence        FLOAT        NOT NULL COMMENT 'Top-1 score in [0, 1]',
    all_scores        TEXT                  COMMENT 'JSON: {"happy":0.92,"sad":0.02,…}',
    image_width       INT                   COMMENT 'Original image width in pixels',
    image_height      INT                   COMMENT 'Original image height in pixels',
    detection_time    FLOAT                 COMMENT 'Inference duration in seconds',
    created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    INDEX idx_emotion   (predicted_emotion),
    INDEX idx_created   (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Emotion detection results';

-- ── Sample verification query ─────────────────────────────────
-- After running the app, check your data with:
--   SELECT id, image_name, predicted_emotion, confidence, created_at
--   FROM emotion_results
--   ORDER BY created_at DESC
--   LIMIT 10;
