export interface CravingStepInfo {
  index: number;
  type: 'init' | 'user_input' | 'agent_response' | 'tool' | 'status' | 'error' | 'system_message';
  state?: string;
  durationSeconds?: number;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    thinkingTokens?: number;
    totalTokens?: number;
  };
  toolName?: string;
  toolParameters?: Record<string, unknown>;
  toolOutput?: string;
  content?: string;
  timestamp?: number;
}

export interface CravingRoundSummary {
  id: string; // e.g. '000001/H2/round-000001'
  epochStr: string;
  epochNum: number;
  task: string;
  roundStr: string;
  roundNum: number;
  status: 'passed' | 'failed' | 'inconclusive' | 'base-failed' | 'running' | 'waiting' | 'unknown';
  activeSeconds: number;
  calls: number;
  error?: {
    message: string;
    type: string;
    traceback?: string;
  };
  hasStreamEvents: boolean;
  hasReport: boolean;
  hasHypothesis: boolean;
  hasExperiment: boolean;
  workspaceDirName?: string;
}

export interface CravingDashboardState {
  campaign: string;
  active: boolean;
  status: string;
  phase: string;
  epoch: number;
  round: number;
  taskIndex: number;
  failures: number;
  lastFailure: {
    message: string;
    type: string;
    traceback?: string;
  } | null;
  failedRound: string | null;
  cumulative: {
    active_seconds?: number;
    compute_active_seconds?: number;
    model_active_seconds?: number;
    declared_configurations?: number;
    declared_feature_sets?: number;
    packets?: number;
    process_attempts?: number;
  };
  nextRetryAt: number;
  updatedAt: number;
  pid: number | null;
  hasStop: boolean;
  goal: {
    schema?: string;
    mandate?: string;
    scope?: string;
    tasks?: {
      id: string;
      title?: string;
      mission?: string;
      enabled?: boolean;
    }[];
    criteria?: Record<string, Record<string, string>>;
    protected_evaluator_sha256?: string;
  } | null;
  bestMetrics?: CravingMetricsData | null;
  recentEvents: {
    index: number;
    at: number;
    kind: string;
    previous_sha256: string | null;
    epoch?: number;
    round?: number;
    reason?: string;
    failure?: { message: string; type: string };
  }[];
}

export interface CravingEpochGroup {
  epochStr: string;
  epochNum: number;
  preregistration: Record<string, unknown> | null;
  tasks: {
    taskId: string;
    rounds: CravingRoundSummary[];
  }[];
}

export interface CravingKpiItem {
  id: string;
  title: string;
  before: string;
  after: string;
  delta: string;
  deltaType: 'positive' | 'negative' | 'neutral';
  bg: string;
  tooltip: string;
  description: string;
}

export interface CravingSimulationWorld {
  id: string;
  name: string;
  category: string;
  brier: string;
  deltaBrier: string;
  auc: string;
  gateOpen: string;
  degraded: string;
  memory: string;
  status: string;
  statusColor: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger';
}

export interface CravingMetricsData {
  sourceRoundId?: string;
  sourceLabel?: string;
  isBest?: boolean;
  timestamp?: number;
  nUsers?: number;
  totalDecisions?: number;
  kpis: CravingKpiItem[];
  simulationWorlds: CravingSimulationWorld[];
  bestConfigId?: string;
  bestAuc?: number;
  deltaBrier?: number;
  rawJson?: Record<string, unknown>;
}

export interface CravingRoundDetails {
  epoch: string;
  task: string;
  round: string;
  modelName: string;
  attemptReport: Record<string, unknown> | null;
  steps: CravingStepInfo[];
  hypothesis: string;
  report: string;
  experimentCode: string;
  metrics?: CravingMetricsData | null;
  bestMetrics?: CravingMetricsData | null;
}
