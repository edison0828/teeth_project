import { fetchPatients } from "../../lib/api";
import UploadForm from "./UploadForm";

export default async function ImageUploadPage() {
  const patients = await fetchPatients();
  return <UploadForm patients={patients.items} />;
}
