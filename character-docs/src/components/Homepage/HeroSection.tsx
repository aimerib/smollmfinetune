import React from 'react';
import Link from '@docusaurus/Link';
import styles from './HeroSection.module.css';

export default function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroBackground}>
        <div className={styles.floatingElement1} />
        <div className={styles.floatingElement2} />
        <div className={styles.floatingElement3} />
      </div>
      
      <div className="container">
        <div className={styles.heroInner}>
          <div className={styles.heroContent}>
            <h1 className={styles.heroTitle}>
              Create AI Characters with
              <span className={styles.heroGradient}> Genuine Personality</span>
            </h1>
            <p className={styles.heroSubtitle}>
              Build believable digital actors with persistent memories, authentic personalities, 
              and the ability to grow through interactions. From concept to conversation in hours, not weeks.
            </p>
            
            <div className={styles.heroCta}>
              <Link
                className={styles.primaryButton}
                to="/docs/setup-guide">
                Get Started
              </Link>
              <Link
                className={styles.secondaryButton}
                to="/docs/">
                Read the Docs
              </Link>
            </div>
            
            <div className={styles.heroFeatures}>
              <div className={styles.feature}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Real-time Responses</span>
              </div>
              <div className={styles.feature}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                <span>Multi-Character Support</span>
              </div>
              <div className={styles.feature}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Consistent Personalities</span>
              </div>
            </div>
          </div>
          
          <div className={styles.heroVisual}>
            <div className={styles.characterCard}>
              <div className={styles.characterAvatar}>
                <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
                  <defs>
                    <linearGradient id="avatarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" style={{stopColor:'#667eea',stopOpacity:1}} />
                      <stop offset="100%" style={{stopColor:'#764ba2',stopOpacity:1}} />
                    </linearGradient>
                  </defs>
                  <circle cx="60" cy="60" r="58" fill="url(#avatarGradient)" opacity="0.1"/>
                  <circle cx="60" cy="45" r="20" fill="url(#avatarGradient)" opacity="0.8"/>
                  <ellipse cx="60" cy="80" rx="18" ry="25" fill="url(#avatarGradient)" opacity="0.8"/>
                </svg>
              </div>
              <div className={styles.characterInfo}>
                <h3>Zara the Space Explorer</h3>
                <div className={styles.traits}>
                  <span className={styles.trait}>Curious</span>
                  <span className={styles.trait}>Adventurous</span>
                  <span className={styles.trait}>Optimistic</span>
                </div>
                <div className={styles.emotions}>
                  <div className={styles.emotion}>
                    <span>Joy</span>
                    <div className={styles.emotionBar}>
                      <div className={styles.emotionFill} style={{width: '75%'}} />
                    </div>
                  </div>
                  <div className={styles.emotion}>
                    <span>Excitement</span>
                    <div className={styles.emotionBar}>
                      <div className={styles.emotionFill} style={{width: '90%'}} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
} 