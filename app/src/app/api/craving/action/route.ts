import { NextRequest, NextResponse } from 'next/server';
import { executeCravingAction } from '../_utils/craving-data';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const action = body.action as 'start' | 'stop' | 'status';
    const campaign = body.campaign || 'gemini';

    if (!action || !['start', 'stop', 'status'].includes(action)) {
      return NextResponse.json({ error: 'Недопустимое действие. Ожидается start, stop или status.' }, { status: 400 });
    }

    const result = await executeCravingAction(action, campaign);
    return NextResponse.json(result);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
