import { notFound } from "next/navigation";

import { fetchAnalysisDetail } from "../../../lib/api";
import { readServerToken } from "../../../lib/server-auth";
import AnalysisDetailWorkspace from "./AnalysisDetailWorkspace";

interface AnalysisPageProps {
  params: { id: string };
}

export default async function AnalysisPage({ params }: AnalysisPageProps) {
  const { id } = params;
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const analysis = await fetchAnalysisDetail(id, token);

  if (!analysis) {
    notFound();
  }

  return <AnalysisDetailWorkspace analysis={analysis} />;
}
