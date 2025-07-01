import type {JSX} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: JSX.Element;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Structured Authoring',
    icon: (
      <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
    description: (
      <>
        Define characters and worlds with structured data, including personality traits, 
        goals, relationships, and world lore. Use control tokens for precise content generation.
      </>
    ),
  },
  {
    title: 'Training Pipeline',
    icon: (
      <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    description: (
      <>
        Complete pipeline from data generation to deployment, including Supervised
        Fine-Tuning (SFT) and Reinforcement Learning (GRPO/PPO) for character alignment.
      </>
    ),
  },
  {
    title: 'Production Inference',
    icon: (
      <svg className={styles.featureSvg} viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    description: (
      <>
        High-performance inference engine with hot-swappable adapters, checkpoint
        sharding for fault-tolerant training, and runtime prompt constructor
        for stateful conversations.
      </>
    ),
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="feature-card">
        <div className="feature-icon">
          {icon}
        </div>
        <div className="text--center padding-horiz--md">
          <Heading as="h3">{title}</Heading>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center margin-bottom--xl">
          <Heading as="h2" className={styles.featuresTitle}>
            Complete Platform for AI Character Development
          </Heading>
          <p className={styles.featuresSubtitle}>
            Tools for building, training, and deploying AI characters with persistent personalities
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
