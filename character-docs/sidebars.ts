import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'doc',
      id: 'README',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: ['setup-guide', 'getting-started', 'client-guide'],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      collapsed: false,
      items: ['core-concepts', 'training-guide', 'TESTING'],
    },
    {
      type: 'category',
      label: 'Advanced',
      collapsed: true,
      items: ['advanced-features', 'deploy', 'nscript', 'checkpoint_sharding', 'triple_head_dpo_training', 'spot_orchestrator', 'directors-view-integration'],
    }
  ],
};

export default sidebars;
