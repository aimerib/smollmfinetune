-- Character Creation Devkit - Development Database Initialization
-- This script sets up the basic database structure for dev-prod environment

-- Ensure the database exists
CREATE DATABASE IF NOT EXISTS dreamcast_dev;

-- Switch to the database
\c dreamcast_dev;

-- Create development user with appropriate permissions
CREATE USER IF NOT EXISTS dreamcast_dev WITH PASSWORD 'dreamcast_dev123';
GRANT ALL PRIVILEGES ON DATABASE dreamcast_dev TO dreamcast_dev;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create basic schema (tables will be created by Alembic migrations)
CREATE SCHEMA IF NOT EXISTS public;
GRANT ALL ON SCHEMA public TO dreamcast_dev;

-- Create monitoring table for development
CREATE TABLE IF NOT EXISTS dev_monitoring (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions to monitoring table
GRANT ALL PRIVILEGES ON TABLE dev_monitoring TO dreamcast_dev;
GRANT ALL PRIVILEGES ON SEQUENCE dev_monitoring_id_seq TO dreamcast_dev;

-- Insert some sample monitoring data
INSERT INTO dev_monitoring (service_name, metric_name, metric_value) VALUES
('platform-api-1', 'response_time_ms', 45.2),
('platform-api-2', 'response_time_ms', 38.7),
('platform-client-1', 'page_load_time_ms', 1205.3),
('platform-client-2', 'page_load_time_ms', 1187.9),
('inference-engine', 'voice_generation_time_ms', 342.1);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_dev_monitoring_service ON dev_monitoring(service_name);
CREATE INDEX IF NOT EXISTS idx_dev_monitoring_timestamp ON dev_monitoring(timestamp);

-- Print completion message
SELECT 'Development database initialized successfully!' AS status; 