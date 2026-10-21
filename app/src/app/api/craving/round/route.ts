import { NextRequest, NextResponse } from 'next/server';
import { getRoundDetails } from '../_utils/craving-data';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const epoch = searchParams.get('epoch');
    const task = searchParams.get('task');
    const round = searchParams.get('round');
    const campaign = searchParams.get('campaign') || 'gemini';

    if (!epoch || !task || !round) {
      return NextResponse.json({ error: 'Параметры epoch, task и round обязательны' }, { status: 400 });
    }

    const details = getRoundDetails(epoch, task, round, campaign);
    if (!details) {
      return NextResponse.json({ error: 'Раунд не найден' }, { status: 404 });
    }

    return NextResponse.json(details);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
