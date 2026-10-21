'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { CravingDashboardState, CravingEpochGroup, CravingRoundDetails, CravingRoundSummary } from '@/types';

export function useCravingDashboard(initialCampaign = 'gemini') {
  const [campaign, setCampaign] = useState(initialCampaign);
  const [state, setState] = useState<CravingDashboardState | null>(null);
  const [epochs, setEpochs] = useState<CravingEpochGroup[]>([]);
  const [selectedRound, setSelectedRound] = useState<CravingRoundSummary | null>(null);
  const [roundDetails, setRoundDetails] = useState<CravingRoundDetails | null>(null);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isActionPending, setIsActionPending] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [selectedTaskFilter, setSelectedTaskFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'ERRORS' | 'SUCCESS'>('ALL');
  const [activeTab, setActiveTab] = useState<'steps' | 'stdout' | 'hypothesis' | 'experiment' | 'report' | 'diagnostics' | 'metrics' | 'guide'>('report');

  const selectedRoundRef = useRef<CravingRoundSummary | null>(null);
  selectedRoundRef.current = selectedRound;

  const stateActiveRef = useRef(false);
  stateActiveRef.current = Boolean(state?.active);

  const [liveActiveSeconds, setLiveActiveSeconds] = useState<number>(0);
  const [liveRoundSeconds, setLiveRoundSeconds] = useState<number>(0);

  // 1. Fetch State
  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(`/api/craving/state?campaign=${encodeURIComponent(campaign)}`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setState(data);
      }
    } catch {
      // ignore network errors
    }
  }, [campaign]);

  // 2. Fetch Epochs & Rounds
  const fetchEpochs = useCallback(async () => {
    try {
      const res = await fetch(`/api/craving/epochs?campaign=${encodeURIComponent(campaign)}`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        const epList: CravingEpochGroup[] = data.epochs || [];
        setEpochs(epList);

        // If no round is selected, pick the latest one
        if (!selectedRoundRef.current && epList.length > 0) {
          for (const ep of epList) {
            for (const t of ep.tasks) {
              if (t.rounds.length > 0) {
                setSelectedRound(t.rounds[0]);
                return;
              }
            }
          }
        }
      }
    } catch {
      // ignore
    }
  }, [campaign]);

  // 3. Fetch Round Details
  const fetchRoundDetails = useCallback(async (round: CravingRoundSummary) => {
    try {
      const url = `/api/craving/round?campaign=${encodeURIComponent(campaign)}&epoch=${encodeURIComponent(round.epochStr)}&task=${encodeURIComponent(round.task)}&round=${encodeURIComponent(round.roundStr)}`;
      const res = await fetch(url, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setRoundDetails(data);
      }
    } catch {
      // ignore
    }
  }, [campaign]);

  // 4. Fetch Terminal Logs
  const fetchTerminalLogs = useCallback(async () => {
    try {
      const res = await fetch(`/api/craving/terminal?campaign=${encodeURIComponent(campaign)}&lines=200`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setTerminalLogs(data.lines || []);
      }
    } catch {
      // ignore
    }
  }, [campaign]);

  // Initial load
  useEffect(() => {
    let mounted = true;
    setIsLoading(true);

    Promise.all([fetchState(), fetchEpochs(), fetchTerminalLogs()]).finally(() => {
      if (mounted) setIsLoading(false);
    });

    return () => {
      mounted = false;
    };
  }, [fetchState, fetchEpochs, fetchTerminalLogs]);

  // When selected round changes, fetch its details
  useEffect(() => {
    if (selectedRound) {
      fetchRoundDetails(selectedRound);
    }
  }, [selectedRound, fetchRoundDetails]);

  // Synchronize live timer with state
  useEffect(() => {
    if (typeof state?.cumulative?.active_seconds === 'number') {
      setLiveActiveSeconds(Math.round(state.cumulative.active_seconds));
    }
  }, [state?.cumulative?.active_seconds]);

  useEffect(() => {
    if (selectedRound) {
      setLiveRoundSeconds(Math.round(selectedRound.activeSeconds || 0));
    }
  }, [selectedRound?.id, selectedRound?.activeSeconds]);

  // Live 1-second ticker when running
  useEffect(() => {
    const isRunning = Boolean(state?.active && (state?.status === 'running' || selectedRound?.status === 'running'));
    if (!isRunning) return;

    const timer = setInterval(() => {
      setLiveActiveSeconds((prev) => prev + 1);
      if (selectedRound?.status === 'running') {
        setLiveRoundSeconds((prev) => prev + 1);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [state?.active, state?.status, selectedRound?.status]);

  // Continuous auto-refresh stream (1.5s when active, 3s when idle)
  useEffect(() => {
    const tick = () => {
      fetchState();
      fetchEpochs();
      if (selectedRoundRef.current) {
        fetchRoundDetails(selectedRoundRef.current);
      }
      fetchTerminalLogs();
    };

    const interval = setInterval(tick, stateActiveRef.current ? 1500 : 3000);
    return () => clearInterval(interval);
  }, [fetchState, fetchEpochs, fetchRoundDetails, fetchTerminalLogs]);

  // Action handlers
  const handleStart = async () => {
    setIsActionPending(true);
    setActionMessage('Запуск автономного исследования...');
    try {
      const res = await fetch('/api/craving/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start', campaign }),
      });
      const data = await res.json();
      setActionMessage(data.message || (data.success ? 'Исследование запущено' : 'Ошибка запуска'));
      await fetchState();
      await fetchEpochs();
    } catch (err: unknown) {
      setActionMessage(`Ошибка сети: ${err}`);
    } finally {
      setIsActionPending(false);
      setTimeout(() => setActionMessage(null), 3000);
    }
  };

  const handleStop = async () => {
    setIsActionPending(true);
    setActionMessage('Отправка запроса на приостановку...');
    try {
      const res = await fetch('/api/craving/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop', campaign }),
      });
      const data = await res.json();
      setActionMessage(data.message || (data.success ? 'Приостановка запрошена' : 'Ошибка'));
      await fetchState();
    } catch (err: unknown) {
      setActionMessage(`Ошибка сети: ${err}`);
    } finally {
      setIsActionPending(false);
      setTimeout(() => setActionMessage(null), 3000);
    }
  };

  const handleSelectRound = (round: CravingRoundSummary) => {
    setSelectedRound(round);
  };

  const handleSelectTaskFilter = (task: string) => {
    setSelectedTaskFilter(task);

    if (task === 'ALL') {
      if (!selectedRoundRef.current && epochs.length > 0) {
        for (const ep of epochs) {
          for (const t of ep.tasks) {
            if (t.rounds.length > 0) {
              setSelectedRound(t.rounds[0]);
              return;
            }
          }
        }
      }
    } else {
      // Find first round for this task across epochs
      for (const ep of epochs) {
        const matchingTask = ep.tasks.find((t) => t.taskId === task);
        if (matchingTask && matchingTask.rounds.length > 0) {
          setSelectedRound(matchingTask.rounds[0]);
          return;
        }
      }
      setSelectedRound(null);
    }
  };

  const handleSelectActiveRound = () => {
    setSelectedTaskFilter('ALL');
    for (const ep of epochs) {
      for (const t of ep.tasks) {
        const active = t.rounds.find((r) => r.status === 'running');
        if (active) {
          setSelectedRound(active);
          return;
        }
      }
    }
    // Fallback to first round
    if (epochs.length > 0 && epochs[0].tasks.length > 0 && epochs[0].tasks[0].rounds.length > 0) {
      setSelectedRound(epochs[0].tasks[0].rounds[0]);
    }
  };

  const refreshAll = useCallback(async () => {
    await Promise.all([
      fetchState(),
      fetchEpochs(),
      fetchTerminalLogs(),
      selectedRoundRef.current ? fetchRoundDetails(selectedRoundRef.current) : Promise.resolve(),
    ]);
  }, [fetchState, fetchEpochs, fetchTerminalLogs, fetchRoundDetails]);

  return {
    campaign,
    setCampaign,
    state,
    epochs,
    selectedRound,
    roundDetails,
    terminalLogs,
    isLoading,
    isActionPending,
    actionMessage,
    selectedTaskFilter,
    setSelectedTaskFilter: handleSelectTaskFilter,
    statusFilter,
    setStatusFilter,
    activeTab,
    setActiveTab,
    liveActiveSeconds,
    liveRoundSeconds,
    handleStart,
    handleStop,
    handleSelectRound,
    handleSelectActiveRound,
    refreshAll,
  };
}
