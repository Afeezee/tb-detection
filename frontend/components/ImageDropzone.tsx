"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import clsx from "clsx";

type Props = {
  onFile: (file: File) => void;
  file: File | null;
  disabled?: boolean;
};

export default function ImageDropzone({ onFile, file, disabled }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"] },
    multiple: false,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={clsx(
        "flex h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition",
        isDragActive
          ? "border-clinical-accent bg-clinical-bg"
          : "border-clinical-border bg-clinical-surface hover:bg-clinical-bg",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <input {...getInputProps()} />
      {file ? (
        <>
          <p className="text-sm font-medium text-clinical-ink">{file.name}</p>
          <p className="mt-1 text-xs text-clinical-muted">
            {(file.size / 1024).toFixed(0)} KB · click or drop to replace
          </p>
        </>
      ) : (
        <>
          <p className="text-sm font-medium text-clinical-ink">
            Drop a chest X-ray here, or click to browse
          </p>
          <p className="mt-1 text-xs text-clinical-muted">PNG or JPEG</p>
        </>
      )}
    </div>
  );
}
