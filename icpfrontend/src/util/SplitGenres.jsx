export default function SplitGenres(genres) {
    return genres.split('|').map(genre => genre.trim());
}