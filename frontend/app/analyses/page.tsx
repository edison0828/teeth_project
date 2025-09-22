import AnalysesWorkspace from './AnalysesWorkspace';

import { fetchAnalyses } from '../../lib/api';
import { readServerToken } from '../../lib/server-auth';

export default async function AnalysesPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const analyses = await fetchAnalyses(token);

  return <AnalysesWorkspace analyses={analyses} />;
}
