import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const fmt = (v) => `₹${(v / 1000).toFixed(0)}k`

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#161616',
      border: '1px solid #2a2a2a',
      borderRadius: '6px',
      padding: '8px 12px',
      fontSize: '12px',
      fontFamily: 'JetBrains Mono, monospace',
    }}>
      <div style={{ color: '#555', marginBottom: 2 }}>{label}</div>
      <div style={{ color: '#e8e8e8' }}>₹{payload[0].value.toLocaleString('en-IN')}</div>
    </div>
  )
}

export default function EquityChart({ data }) {
  if (!data?.length) return null

  // Thin to max 150 points for performance
  const step = Math.ceil(data.length / 150)
  const thinned = data.filter((_, i) => i % step === 0 || i === data.length - 1)

  const initial = 100000
  const isProfit = data[data.length - 1].equity >= initial
  const color = isProfit ? '#22c55e' : '#ef4444'

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={thinned} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="date"
            tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: 'Inter' }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#3a3a3a', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={fmt}
            width={48}
          />
          <ReferenceLine y={initial} stroke="#2a2a2a" strokeDasharray="3 3" />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="equity"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: color, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
