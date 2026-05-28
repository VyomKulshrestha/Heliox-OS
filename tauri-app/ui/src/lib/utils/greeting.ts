export function getJarvisGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 6) {
    return "Good evening. Heliox OS is online.";
  }
  if (hour < 12) {
    return "Good morning. Heliox OS is ready.";
  }
  if (hour < 17) {
    return "Good afternoon. Heliox OS at your service.";
  }
  if (hour < 21) {
    return "Good evening. Heliox OS is ready.";
  }
  return "Burning the midnight oil?";
}