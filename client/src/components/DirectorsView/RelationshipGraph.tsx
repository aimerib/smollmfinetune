/**
 * RelationshipGraph Component
 * 
 * Interactive D3.js-based visualization of character relationships
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import styled from '@emotion/styled';
import { RelationshipNode, RelationshipEdge, AffinityFilter } from '../../types/relationships';

const GraphContainer = styled.div`
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 8px;
  position: relative;
  overflow: hidden;
`;

const Tooltip = styled.div`
  position: absolute;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.9);
  color: white;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 100;
  
  &.visible {
    opacity: 1;
  }
`;

interface RelationshipGraphProps {
  nodes: RelationshipNode[];
  edges: RelationshipEdge[];
  onNodeSelect?: (nodeId: string) => void;
  onEdgeSelect?: (edge: RelationshipEdge) => void;
  affinityFilter?: AffinityFilter;
}

export const RelationshipGraph: React.FC<RelationshipGraphProps> = ({
  nodes,
  edges,
  onNodeSelect,
  onEdgeSelect,
  affinityFilter = { min: -1, max: 1 }
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Get dominant emotion for a node
  const getDominantEmotion = (emotionalState: RelationshipNode['emotional_state']): string => {
    let maxEmotion = 'neutral';
    let maxValue = 0;
    
    Object.entries(emotionalState).forEach(([emotion, value]) => {
      if (value && value > maxValue) {
        maxValue = value;
        maxEmotion = emotion;
      }
    });
    
    return maxEmotion;
  };

  // Get color based on emotion
  const getEmotionColor = (emotion: string): string => {
    const emotionColors: Record<string, string> = {
      happy: '#4ade80',
      sad: '#60a5fa',
      angry: '#f87171',
      fearful: '#c084fc',
      surprised: '#fbbf24',
      disgusted: '#a78bfa',
      nervous: '#fb923c',
      neutral: '#9ca3af'
    };
    return emotionColors[emotion] || emotionColors.neutral;
  };

  // Get color for relationship status
  const getRelationshipColor = (status: string, affinity: number): string => {
    if (affinity > 0) {
      return affinity > 0.5 ? '#22c55e' : '#86efac'; // Green shades for positive
    } else if (affinity < 0) {
      return affinity < -0.5 ? '#ef4444' : '#fca5a5'; // Red shades for negative
    }
    return '#9ca3af'; // Gray for neutral
  };

  useEffect(() => {
    if (!svgRef.current) return;

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    // Filter edges based on affinity
    const filteredEdges = edges.filter(edge => 
      edge.affinity >= affinityFilter.min && edge.affinity <= affinityFilter.max
    );

    // Set up SVG
    const svg = d3.select(svgRef.current)
      .attr('width', dimensions.width)
      .attr('height', dimensions.height);

    // Create groups for layers
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Create edge elements
    const edgeGroup = g.append('g').attr('class', 'edges');
    
    const edgeElements = edgeGroup.selectAll('.edge')
      .data(filteredEdges)
      .enter()
      .append('line')
      .attr('class', edge => `edge relationship-${edge.affinity > 0 ? 'positive' : 'negative'}`)
      .attr('data-testid', d => `edge-${d.source}-${d.target}`)
      .attr('data-status', d => d.status)
      .attr('x1', d => {
        const sourceNode = nodes.find(n => n.id === d.source);
        return sourceNode?.position.x || 0;
      })
      .attr('y1', d => {
        const sourceNode = nodes.find(n => n.id === d.source);
        return sourceNode?.position.y || 0;
      })
      .attr('x2', d => {
        const targetNode = nodes.find(n => n.id === d.target);
        return targetNode?.position.x || 0;
      })
      .attr('y2', d => {
        const targetNode = nodes.find(n => n.id === d.target);
        return targetNode?.position.y || 0;
      })
      .attr('stroke', d => getRelationshipColor(d.status, d.affinity))
      .attr('stroke-width', d => Math.abs(d.affinity) * 10) // Thickness based on affinity strength
      .attr('opacity', d => Math.abs(d.affinity) * 0.3 + 0.3) // Opacity based on affinity
      .style('stroke', d => getRelationshipColor(d.status, d.affinity))
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        // Show tooltip
        if (tooltipRef.current) {
          tooltipRef.current.innerHTML = `
            <div><strong>${d.source} → ${d.target}</strong></div>
            <div>Status: ${d.status}</div>
            <div>Affinity: ${d.affinity.toFixed(2)}</div>
            <div>Memory Significance: ${d.memory_significance.toFixed(1)}</div>
            <div>${d.interaction_count} interactions</div>
          `;
          tooltipRef.current.style.left = `${event.pageX + 10}px`;
          tooltipRef.current.style.top = `${event.pageY - 10}px`;
          tooltipRef.current.classList.add('visible');
        }
      })
      .on('mouseleave', function() {
        if (tooltipRef.current) {
          tooltipRef.current.classList.remove('visible');
        }
      })
      .on('click', function(event, d) {
        event.stopPropagation();
        if (onEdgeSelect) {
          onEdgeSelect(d);
        }
      });

    // Create node elements
    const nodeGroup = g.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.position.x}, ${d.position.y})`);

    // Add circles for nodes
    nodeElements.append('circle')
      .attr('data-testid', d => `node-${d.id}`)
      .attr('data-emotion', d => getDominantEmotion(d.emotional_state))
      .attr('r', d => d.size / 2)
      .attr('fill', d => getEmotionColor(getDominantEmotion(d.emotional_state)))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (onNodeSelect) {
          onNodeSelect(d.id);
        }
      });

    // Add labels
    nodeElements.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('font-size', 12)
      .attr('fill', '#fff')
      .style('pointer-events', 'none');

    // Animation for new relationships
    edgeElements
      .filter(function() {
        const edge = d3.select(this);
        return !edge.classed('existing');
      })
      .classed('relationship-forming', true)
      .classed('existing', true);

  }, [nodes, edges, affinityFilter, dimensions, onNodeSelect]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current?.parentElement) {
        const { width, height } = svgRef.current.parentElement.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <GraphContainer>
      <svg ref={svgRef} />
      <Tooltip ref={tooltipRef} role="tooltip" />
    </GraphContainer>
  );
}; 