const url = `https://image.tmdb.org/t/p/w500`;

export default function getImage(image: string): string {
  return `${url}${image}`;
}