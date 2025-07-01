# R7-1 Character DNA Breeding System
Status: **Todo**
Ring: R7
Created: 2025-01-14
---

## Goal
Implement a genetic breeding system that allows creators to combine two characters' traits to produce offspring with emergent personalities, creating natural character family trees and evolutionary narratives.

## Context
While creators can craft individual characters masterfully, they lack tools for exploring character relationships through lineage. This system would allow characters to have children, siblings, and ancestors, with traits that naturally flow through generations. This opens entirely new narrative possibilities and creates a marketplace for "breeding stock" characters with desirable traits.

## Acceptance Criteria

### Core Breeding Mechanics
- [ ] Create `CharacterGenetics` class that encodes Big Five traits as "genes"
- [ ] Implement crossover algorithm that combines parent traits with variance
- [ ] Add mutation system for emergent trait combinations (5-10% chance)
- [ ] Support both sexual (two parents) and asexual (one parent + mutation) reproduction
- [ ] Maintain trait constraints (all values 0-1 range, meaningful combinations)

### Inheritance System
- [ ] Dominant/recessive trait modeling for personality aspects
- [ ] Goal inheritance with generational drift
- [ ] Relationship predispositions (e.g., "naturally trusting" as heritable)
- [ ] Memory compatibility (offspring can access ancestral memories with decay)
- [ ] Physical appearance description inheritance and combination

### Family Tree Infrastructure
- [ ] Database schema for character lineages and relationships
- [ ] Visual family tree component using D3.js or similar
- [ ] Generation tracking and ancestry queries
- [ ] Breeding history and genetic diversity metrics
- [ ] Export family trees as JSON or GraphML

### UI Integration
- [ ] "Breeding Chamber" page in character management
- [ ] Partner selection interface with compatibility scoring
- [ ] Trait preview for potential offspring
- [ ] Breeding animation/ritual for user engagement
- [ ] Family tree browser with relationship details

### Genetic Algorithms
- [ ] Fitness functions for trait optimization
- [ ] Multi-generation breeding simulations
- [ ] Trait stability analysis across generations
- [ ] Hybrid vigor modeling (crossbreeds stronger than purebreds)

## Implementation Notes
```text
• Genetics Encoding:
  - Each Big Five trait = 2 genes (dominant/recessive)
  - Goals encoded as "gene sequences" with inheritance probabilities
  - Mutation rates increase with "magical" or "sci-fi" world types
  
• Breeding Rules:
  - Compatible species/type checking
  - Minimum "maturity" (training completeness) before breeding
  - Cooldown periods to prevent spam
  - "Breeding licenses" for premium features
  
• Technical Architecture:
  - Extend CharacterCore with genetic_code field
  - New table: character_lineages (parent_ids, child_id, generation)
  - Breeding transaction system to prevent conflicts
  - Background job for complex multi-generation simulations
```

## Checklist / Steps
1. Design genetic encoding system for personality traits
2. Implement core crossover and mutation algorithms
3. Create database schema for lineages
4. Build breeding compatibility calculator
5. Develop trait inheritance logic
6. Create family tree visualization component
7. Implement breeding UI with preview
8. Add breeding history tracking
9. Create multi-generation simulation tools
10. Write comprehensive genetic diversity analytics
11. Build export system for family narratives
12. Test with various character combinations

## References
- Depends on: R1-2 (CharacterCore structure), R1-5 (Personality traits)
- Enhances: R3-0.9 (Character Selection - can now select relatives)
- Enables: Character Dynasty narratives, Genetic marketplace