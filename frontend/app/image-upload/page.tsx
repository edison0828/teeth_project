import { fetchPatients } from "../../lib/api";
import { readServerToken } from "../../lib/server-auth";
import UploadForm from "./UploadForm";

export default async function ImageUploadPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const patients = await fetchPatients(undefined, token);
  return <UploadForm patients={patients.items} />;
}
