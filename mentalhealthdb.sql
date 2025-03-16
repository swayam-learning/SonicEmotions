-- Active: 1740077526153@@127.0.0.1@3306@mentalhealthdb

CREATE DATABASE MentalHealthDB;
USE MentalHealthDB;

CREATE TABLE reddit_posts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    post_id VARCHAR(20) UNIQUE,
    title TEXT,
    body TEXT,
    upvotes INT,
    created_at DATETIME
);

CREATE TABLE reddit_comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    post_id VARCHAR(20) UNIQUE,
    comments TEXT,
    upvotes INT,
    created_at DATETIME,
    Foreign Key (post_id) REFERENCES reddit_posts(post_id) ON DELETE CASCADE
);

SELECT * FROM reddit_posts;
SELECT * FROM reddit_comments;
drop DATABASE mentalhealthdb;
