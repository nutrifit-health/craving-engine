'use client';

import { useState } from 'react';
import { useCravingDashboard } from '@/hooks/useCravingDashboard';
import { CravingNavbar } from './CravingNavbar';
import { CravingSidebar } from './CravingSidebar';
import { CravingRoundsList } from './CravingRoundsList';
import { CravingTerminal } from './CravingTerminal';
import { CravingEventsModal } from './CravingEventsModal';

export function CravingDashboard() {
  const {
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
    setSelectedTaskFilter,
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
  } = useCravingDashboard('gemini');

  const [isEventsModalOpen, setIsEventsModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className='w-full min-w-0 min-h-screen bg-background text-foreground flex flex-col'>
      {/* Fixed Top Navbar with live ticking seconds and sidebar toggle */}
      <CravingNavbar
        state={state}
        campaign={campaign}
        onCampaignChange={setCampaign}
        onStart={handleStart}
        onStop={handleStop}
        isActionPending={isActionPending}
        actionMessage={actionMessage}
        liveActiveSeconds={liveActiveSeconds}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
      />

      {/* Main Body: Top padding pt-[74px] for fixed navbar */}
      <div className='flex flex-1 w-full min-w-0 pt-[74px]'>
        {/* Navigation Sidebar (collapsible via burger button) */}
        {isSidebarOpen && (
          <CravingSidebar
            state={state}
            epochs={epochs}
            selectedTaskFilter={selectedTaskFilter}
            onSelectTaskFilter={setSelectedTaskFilter}
            onOpenEvents={() => setIsEventsModalOpen(true)}
            onSelectTab={setActiveTab}
          />
        )}

        {/* Content: Split into Left (Iterations/Steps) and Right (Terminal/Output) */}
        <main className='flex-1 flex min-w-0 overflow-hidden'>
          {/* Left Column: Chronology, Epochs, Rounds */}
          <div className='w-80 md:w-84 xl:w-96 shrink-0 min-w-0'>
            <CravingRoundsList
              epochs={epochs}
              selectedRound={selectedRound}
              onSelectRound={handleSelectRound}
              selectedTaskFilter={selectedTaskFilter}
              statusFilter={statusFilter}
              onStatusFilterChange={setStatusFilter}
              onResetTaskFilter={handleSelectActiveRound}
            />
          </div>

          {/* Right Column: Execution Terminal, Gemini Steps, Errors */}
          <div className='flex-1 min-w-0'>
            <CravingTerminal
              state={state}
              selectedRound={selectedRound}
              roundDetails={roundDetails}
              terminalLogs={terminalLogs}
              activeTab={activeTab}
              onTabChange={setActiveTab}
              isLoading={isLoading}
              onSelectActiveRound={handleSelectActiveRound}
              liveRoundSeconds={liveRoundSeconds}
              selectedTaskFilter={selectedTaskFilter}
            />
          </div>
        </main>
      </div>

      {/* Events Modal */}
      <CravingEventsModal
        isOpen={isEventsModalOpen}
        onClose={() => setIsEventsModalOpen(false)}
        events={state?.recentEvents || []}
      />
    </div>
  );
}
