# R7-1: Character DNA Breeding System
Status: **Todo**
Ring: R7
Created: 2025-01-20
---

## Goal
Implement a genetic breeding system within the Dreamcast platform that allows creators to combine character traits and produce offspring with emergent personalities, creating natural character family trees and evolutionary narratives for rich world building.

## Context
**Dreamcast Platform Vision**: This feature exemplifies the platform as both creative devkit and immersive console experience.

While creators can craft individual characters masterfully, they lack tools for exploring character relationships through lineage within the unified React+FastAPI platform. This system allows characters to have children, siblings, and ancestors, with traits that naturally flow through generations. This opens entirely new narrative possibilities and creates a marketplace for "breeding stock" characters with desirable traits.

**Platform Integration**: Full React UI with real-time breeding visualization, FastAPI backend for genetic algorithms, and potential cartridge export for family lineages.

## Acceptance Criteria

### Core Breeding Mechanics (React + FastAPI)
- [ ] **Genetic Engine API**: FastAPI endpoints for genetic crossover and mutation algorithms
- [ ] **React Breeding Interface**: Interactive breeding chamber with real-time trait preview
- [ ] **Big Five Genetics**: Encode personality traits as "genes" with dominant/recessive modeling
- [ ] **Crossover Algorithm**: Combine parent traits with variance and emergent properties
- [ ] **Mutation System**: 5-10% chance for emergent trait combinations
- [ ] **Reproduction Types**: Support both sexual (two parents) and asexual (one parent + mutation)

### Inheritance System (Platform Integration)
- [ ] **Trait Inheritance**: Dominant/recessive modeling for personality aspects
- [ ] **Goal Evolution**: Goal inheritance with generational drift and adaptation
- [ ] **Relationship Genetics**: Heritable predispositions (e.g., "naturally trusting")
- [ ] **Memory Compatibility**: Offspring access ancestral memories with decay over generations
- [ ] **Physical Inheritance**: Appearance description combination and evolution
- [ ] **Voice Genetics**: Inherit and blend voice characteristics from parents

### Family Tree Infrastructure (React Visualization)
- [ ] **Database Schema**: Character lineages, relationships, and genetic history
- [ ] **Interactive Family Tree**: React component using D3.js for immersive visualization
- [ ] **Generation Tracking**: Ancestry queries and lineage exploration
- [ ] **Genetic Diversity**: Metrics and analytics for breeding optimization
- [ ] **Export System**: Family trees as JSON, GraphML, or cartridge data

### Dreamcast Console Experience
- [ ] **Breeding Chamber**: Immersive React interface with console-quality UX
- [ ] **Partner Selection**: Compatibility scoring with visual trait comparison
- [ ] **Trait Preview**: Real-time visualization of potential offspring
- [ ] **Breeding Ritual**: Engaging animation/ceremony for user immersion
- [ ] **Family Browser**: Rich relationship exploration with character details
- [ ] **Genetic Analytics**: Dashboard for breeding strategy and optimization

### Cartridge Integration (Mini Experience)
- [ ] **Family Lineage Export**: Export family trees as portable cartridge data
- [ ] **Genetic Summary**: Compressed genetic information for cartridge characters
- [ ] **Lineage Stories**: Mini-narratives about character ancestry for cartridge users
- [ ] **Breeding History**: Simplified family tree for cartridge character context

### Advanced Genetic Features
- [ ] **Fitness Functions**: Trait optimization algorithms for selective breeding
- [ ] **Multi-Generation Simulation**: Simulate breeding across multiple generations
- [ ] **Trait Stability**: Analysis of genetic stability across generations
- [ ] **Hybrid Vigor**: Crossbreeds stronger than purebreds modeling
- [ ] **Genetic Marketplace**: Platform for trading breeding stock characters

## Implementation Notes
```text
• Platform Architecture:
  - FastAPI genetic engine with async breeding algorithms
  - React breeding interface with real-time trait visualization
  - WebSocket updates for breeding progress and results
  - Database integration for family lineage storage
  
• Genetics Encoding:
  - Each Big Five trait = 2 genes (dominant/recessive)
  - Goals encoded as "gene sequences" with inheritance probabilities
  - Mutation rates increase with "magical" or "sci-fi" world types
  - Voice characteristics encoded as genetic parameters
  
• Breeding Rules:
  - Compatible species/type checking via API validation
  - Minimum "maturity" (training completeness) before breeding
  - Cooldown periods to prevent spam breeding
  - "Breeding licenses" for premium platform features
  
• Dreamcast Experience:
  - Console-quality UI with smooth animations and transitions
  - Rich visual feedback for genetic combinations
  - Immersive breeding ceremony with character interaction
  - Family tree exploration with zoom and filter capabilities
  
• Cartridge Integration:
  - Compress family lineage data for portable export
  - Include genetic summaries in character cartridge metadata
  - Enable mini-family tree viewing in cartridge players
```

## TDD Instructions
- **Genetic Algorithm Tests**: Test crossover, mutation, and inheritance algorithms
- **React Component Tests**: Test breeding interface and family tree visualization
- **API Tests**: Test FastAPI endpoints for breeding and lineage management
- **Integration Tests**: Test end-to-end breeding workflow from selection to offspring
- **Performance Tests**: Test breeding simulation performance with large family trees

## Checklist / Steps
1. **Design genetic encoding system** for personality traits and characteristics
2. **Implement FastAPI genetic engine** with crossover and mutation algorithms
3. **Create React breeding interface** with trait visualization and selection
4. **Build database schema** for character lineages and relationships
5. **Develop breeding compatibility calculator** with trait analysis
6. **Implement trait inheritance logic** with dominant/recessive modeling
7. **Create interactive family tree component** with D3.js visualization
8. **Build breeding chamber UI** with immersive console experience
9. **Add real-time breeding progress** with WebSocket updates
10. **Implement genetic diversity analytics** and optimization tools
11. **Create family lineage export system** for cartridge integration
12. **Add breeding history tracking** and genealogy features
13. **Build genetic marketplace** for character trading
14. **Create multi-generation simulation** tools for advanced breeding
15. **Test with various character combinations** and edge cases
16. **Add comprehensive documentation** for breeding system usage

## References
- Depends on: R6-3.1, R6-3.2, R6-3.3 (Architecture Migration), R1-2 (CharacterCore), R1-5 (Personality traits)
- Enhances: R3-0.9 (Character Selection - can now select relatives)
- Enables: Character Dynasty narratives, Genetic marketplace, Advanced world building
- Architecture: See overview.mdc architecture diagram
- Platform Integration: React+FastAPI unified architecture