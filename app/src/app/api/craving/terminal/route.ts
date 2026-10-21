import { NextRequest, NextResponse } from 'next/server';
import { getStdoutLogs } from '../_utils/craving-data';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const campaign = searchParams.get('campaign') || 'gemini';
    const linesCount = parseInt(searchParams.get('lines') || '250', 10);
    const logs = getStdoutLogs(campaign, linesCount);
    return NextResponse.json({ lines: logs });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
