/** A trade's colour by direction and outcome, so a losing long never reads as a winner: long
 *  bright green when it made money, red when it lost; short light blue when it made money,
 *  fuchsia red when it lost. Shared by the day chart and the P&L calendar. */
export const HOLD = {
  longWin: { fill: '#22e55e', opacity: 0.32, label: 'long, profit' },
  longLoss: { fill: '#ef4444', opacity: 0.38, label: 'long, loss' },
  shortWin: { fill: '#7dd3fc', opacity: 0.38, label: 'short, profit' },
  shortLoss: { fill: '#f0167e', opacity: 0.38, label: 'short, loss' },
} as const

export const holdShade = (pos: number, ret: number) =>
  pos > 0 ? (ret >= 0 ? HOLD.longWin : HOLD.longLoss) : ret >= 0 ? HOLD.shortWin : HOLD.shortLoss
