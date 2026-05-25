import { useEffect, useCallback, useState, useRef } from 'react'
import Graph from 'graphology'
import { SigmaContainer, useLoadGraph, useRegisterEvents, useSigma } from '@react-sigma/core'
import '@react-sigma/core/lib/style.css'
import forceAtlas2 from 'graphology-layout-forceatlas2'

interface GraphNode {
  id: string
  label: string
  type: string
  path: string
  link_count: number
  community: number
}

interface GraphEdge {
  source: string
  target: string
  weight: number
}

interface CommunityInfo {
  id: number
  node_count: number
  cohesion: number
  top_nodes: string[]
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  communities: CommunityInfo[]
}

const NODE_TYPE_COLORS: Record<string, string> = {
  entity: '#60a5fa',
  concept: '#c084fc',
  source: '#fb923c',
  query: '#4ade80',
  synthesis: '#f87171',
  overview: '#facc15',
  comparison: '#2dd4bf',
  other: '#94a3b8',
}

const COMMUNITY_COLORS = [
  '#60a5fa', '#4ade80', '#fb923c', '#c084fc', '#f87171',
  '#2dd4bf', '#facc15', '#f472b6', '#a78bfa', '#38bdf8',
  '#34d399', '#fbbf24',
]

const BASE_NODE_SIZE = 8
const MAX_NODE_SIZE = 28

function nodeColor(type: string): string {
  return NODE_TYPE_COLORS[type] ?? NODE_TYPE_COLORS.other
}

function nodeSize(linkCount: number, maxLinks: number): number {
  if (maxLinks === 0) return BASE_NODE_SIZE
  const ratio = linkCount / maxLinks
  return BASE_NODE_SIZE + Math.sqrt(ratio) * (MAX_NODE_SIZE - BASE_NODE_SIZE)
}

// Cache positions to avoid re-layout on re-renders
const positionCache = new Map<string, { x: number; y: number }>()
let lastLayoutDataKey = ''

function GraphLoader({ nodes, edges, colorMode }: { nodes: GraphNode[]; edges: GraphEdge[]; colorMode: 'type' | 'community' }) {
  const loadGraph = useLoadGraph()

  useEffect(() => {
    const dataKey = nodes.map((n) => n.id).sort().join(',') + '|' + edges.length
    const needsLayout = dataKey !== lastLayoutDataKey

    const graph = new Graph()
    const maxLinks = Math.max(...nodes.map((n) => n.link_count), 1)

    for (const node of nodes) {
      const cached = positionCache.get(node.id)
      const color = colorMode === 'community'
        ? COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length]
        : nodeColor(node.type)
      graph.addNode(node.id, {
        x: cached?.x ?? Math.random() * 100,
        y: cached?.y ?? Math.random() * 100,
        size: nodeSize(node.link_count, maxLinks),
        color,
        label: node.label,
        nodeType: node.type,
        nodePath: node.path,
        community: node.community,
      })
    }

    const maxWeight = Math.max(...edges.map((e) => e.weight), 1)

    for (const edge of edges) {
      if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
        const edgeKey = `${edge.source}->${edge.target}`
        if (!graph.hasEdge(edgeKey) && !graph.hasEdge(`${edge.target}->${edge.source}`)) {
          const normalizedWeight = edge.weight / maxWeight
          const size = 0.5 + normalizedWeight * 3.5
          const alpha = Math.round(40 + normalizedWeight * 180)
          const color = `rgba(100,116,139,${alpha / 255})`
          graph.addEdgeWithKey(edgeKey, edge.source, edge.target, {
            color,
            size,
            weight: edge.weight,
          })
        }
      }
    }

    if (needsLayout && nodes.length > 1) {
      const settings = forceAtlas2.inferSettings(graph)
      forceAtlas2.assign(graph, {
        iterations: 150,
        settings: {
          ...settings,
          gravity: 1,
          scalingRatio: 2,
          strongGravityMode: true,
          barnesHutOptimize: nodes.length > 50,
        },
      })
      lastLayoutDataKey = dataKey

      graph.forEachNode((nodeId, attrs) => {
        positionCache.set(nodeId, { x: attrs.x, y: attrs.y })
      })
    }

    loadGraph(graph)
  }, [loadGraph, nodes, edges, colorMode])

  return null
}

function EventHandler({ onNodeClick }: { onNodeClick: (nodeId: string, path: string) => void }) {
  const registerEvents = useRegisterEvents()
  const sigma = useSigma()

  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => {
        const graph = sigma.getGraph()
        const attrs = graph.getNodeAttributes(node)
        onNodeClick(node, attrs.nodePath || '')
      },
      enterNode: ({ node }) => {
        const container = sigma.getContainer()
        container.style.cursor = 'pointer'
        const graph = sigma.getGraph()
        graph.setNodeAttribute(node, 'hovering', true)
        const neighbors = new Set(graph.neighbors(node))
        neighbors.add(node)
        graph.forEachNode((n) => {
          if (!neighbors.has(n)) graph.setNodeAttribute(n, 'dimmed', true)
        })
        graph.forEachEdge((e, _attrs, source, target) => {
          if (source !== node && target !== node) {
            graph.setEdgeAttribute(e, 'dimmed', true)
          } else {
            graph.setEdgeAttribute(e, 'highlighted', true)
          }
        })
        sigma.refresh()
      },
      leaveNode: () => {
        const container = sigma.getContainer()
        container.style.cursor = 'default'
        const graph = sigma.getGraph()
        graph.forEachNode((n) => {
          graph.removeNodeAttribute(n, 'hovering')
          graph.removeNodeAttribute(n, 'dimmed')
        })
        graph.forEachEdge((e) => {
          graph.removeEdgeAttribute(e, 'dimmed')
          graph.removeEdgeAttribute(e, 'highlighted')
        })
        sigma.refresh()
      },
    })
  }, [registerEvents, sigma, onNodeClick])

  return null
}

function ZoomControls() {
  const sigma = useSigma()

  return (
    <div className="absolute top-3 right-3 flex flex-col gap-1">
      <button
        onClick={() => sigma.getCamera().animatedZoom({ duration: 200 })}
        className="h-7 w-7 flex items-center justify-center rounded bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 shadow text-sm"
      >
        +
      </button>
      <button
        onClick={() => sigma.getCamera().animatedUnzoom({ duration: 200 })}
        className="h-7 w-7 flex items-center justify-center rounded bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 shadow text-sm"
      >
        −
      </button>
      <button
        onClick={() => sigma.getCamera().animatedReset({ duration: 300 })}
        className="h-7 w-7 flex items-center justify-center rounded bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 shadow text-xs"
      >
        ⊡
      </button>
    </div>
  )
}

export function GraphView() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [], communities: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [colorMode, setColorMode] = useState<'type' | 'community'>('type')
  const [sigmaKey, setSigmaKey] = useState(0)

  useEffect(() => {
    fetch('/api/v1/wiki/graph')
      .then(res => res.json())
      .then(data => {
        setGraphData(data)
        setLoading(false)
        setSigmaKey(k => k + 1)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const handleNodeClick = useCallback((nodeId: string, path: string) => {
    const node = graphData.nodes.find(n => n.id === nodeId)
    if (node) setSelectedNode(node)
  }, [graphData.nodes])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">加载图谱中...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">加载失败: {error}</div>
      </div>
    )
  }

  if (graphData.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">暂无图谱数据</div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full">
      {/* Header */}
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <button
          onClick={() => setColorMode(colorMode === 'type' ? 'community' : 'type')}
          className="px-3 py-1.5 text-sm rounded bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"
        >
          {colorMode === 'type' ? '社区模式' : '类型模式'}
        </button>
        <button
          onClick={() => setSigmaKey(k => k + 1)}
          className="px-3 py-1.5 text-sm bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
        >
          重新布局
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-white/90 dark:bg-gray-800/90 p-3 rounded-lg text-sm">
        <div className="font-medium mb-2">节点类型</div>
        <div className="grid grid-cols-2 gap-1">
          {Object.entries(NODE_TYPE_COLORS).filter(([k]) => k !== 'other').map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="capitalize">{type}</span>
            </div>
          ))}
        </div>
        {colorMode === 'community' && (
          <>
            <div className="font-medium mt-3 mb-2">社区 (共{graphData.communities.length}个)</div>
            <div className="max-h-32 overflow-y-auto">
              {graphData.communities.slice(0, 5).map((comm, i) => (
                <div key={comm.id} className="flex items-center gap-1.5 text-xs">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COMMUNITY_COLORS[i % COMMUNITY_COLORS.length] }} />
                  <span>{comm.top_nodes[0] || `社区${comm.id}`} ({comm.node_count})</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Selected node info */}
      {selectedNode && (
        <div className="absolute bottom-4 right-4 z-10 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg w-64">
          <div className="flex justify-between items-start">
            <div>
              <div className="font-medium">{selectedNode.label}</div>
              <div className="text-xs text-gray-500 mt-1">类型: {selectedNode.type}</div>
              <div className="text-xs text-gray-500">社区: {selectedNode.community}</div>
              <div className="text-xs text-gray-500">链接数: {selectedNode.link_count}</div>
            </div>
            <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-gray-600">✕</button>
          </div>
          <div className="text-xs text-gray-400 mt-2 truncate">{selectedNode.path}</div>
        </div>
      )}

      {/* Graph canvas */}
      <SigmaContainer
        key={sigmaKey}
        style={{ width: '100%', height: '100%' }}
        settings={{
          renderEdgeLabels: false,
          defaultEdgeColor: '#cbd5e1',
          defaultNodeColor: '#94a3b8',
          labelSize: 13,
          labelWeight: 'bold',
          labelColor: { color: '#1e293b' },
          labelDensity: 0.3,
          labelRenderedSizeThreshold: 8,
          stagePadding: 30,
          nodeReducer: (_node, attrs) => {
            const result = { ...attrs }
            if (attrs.hovering) {
              result.size = (attrs.size ?? BASE_NODE_SIZE) * 1.4
              result.zIndex = 10
              result.forceLabel = true
            }
            if (attrs.dimmed) {
              result.color = '#e2e8f0'
              result.label = ''
              result.size = (attrs.size ?? BASE_NODE_SIZE) * 0.6
            }
            return result
          },
          edgeReducer: (_edge, attrs) => {
            const result = { ...attrs }
            if (attrs.dimmed) {
              result.color = '#f1f5f9'
              result.size = 0.3
            }
            if (attrs.highlighted) {
              result.color = '#1e293b'
              result.size = Math.max(2, (attrs.size ?? 1) * 1.5)
            }
            return result
          },
        }}
      >
        <GraphLoader nodes={graphData.nodes} edges={graphData.edges} colorMode={colorMode} />
        <EventHandler onNodeClick={handleNodeClick} />
        <ZoomControls />
      </SigmaContainer>
    </div>
  )
}