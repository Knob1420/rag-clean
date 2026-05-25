/**
 * SSE Handler Web Worker
 *
 * 专门负责解析 SSE 事件，不受主线程阻塞影响。
 * 将完整事件通过 postMessage 发送回主线程。
 */

let eventBuffer = { event: '', data: '' }

self.onmessage = (e: MessageEvent) => {
  const { type, data } = e.data

  if (type === 'chunk') {
    // 处理接收到的文本 chunk
    const text = data as string
    const lines = text.split('\n')

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventBuffer.event = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        eventBuffer.data = line.slice(6).trim()
      } else if (line.trim() === '' && eventBuffer.event && eventBuffer.data) {
        // 完整事件，发送到主线程
        self.postMessage({
          type: 'event',
          eventType: eventBuffer.event,
          data: eventBuffer.data,
        })
        eventBuffer = { event: '', data: '' }
      }
    }
  } else if (type === 'flush') {
    // 处理完所有数据后的刷新
    if (eventBuffer.event && eventBuffer.data) {
      self.postMessage({
        type: 'event',
        eventType: eventBuffer.event,
        data: eventBuffer.data,
      })
      eventBuffer = { event: '', data: '' }
    }
    self.postMessage({ type: 'done' })
  }
}