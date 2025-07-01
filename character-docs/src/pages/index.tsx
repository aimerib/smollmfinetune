import type {JSX} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HeroSection from '@site/src/components/Homepage/HeroSection';

import styles from './index.module.css';

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Platform for creating, training, and deploying AI characters with persistent personalities">
      <HeroSection />
      <main>
        <HomepageFeatures />
      </main>
      <section className={styles.ctaSection}>
        <div className="container">
          <Heading as="h2" className={styles.ctaTitle}>Get Started</Heading>
          <p className={styles.ctaDescription}>
            Follow our documentation to set up the platform and create your first AI character.
          </p>
          <div className={styles.ctaButtons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/getting-started">
              Getting Started
            </Link>
            <Link
              className="button button--outline button--lg"
              to="/docs/setup-guide">
              Setup Guide
            </Link>
          </div>
        </div>
      </section>
    </Layout>
  );
}
