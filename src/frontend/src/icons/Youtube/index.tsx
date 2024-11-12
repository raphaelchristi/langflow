import React, { forwardRef } from "react";
import YouTubeIcon from "./youtube";

export const YouTubeSvgIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <YouTubeIcon ref={ref} {...props} />;
});