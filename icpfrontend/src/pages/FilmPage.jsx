import { Container, Typography, Rating, Paper } from '@mui/material';
import StarIcon from '@mui/icons-material/Star';
import Box from '@mui/material/Box'; 
import useFetchData from '../hooks/useFetchData';
import { useParams } from 'react-router-dom';
import LoadingInPage from '../components/LoadingInPage';
import SwipeableTemporaryDrawer from '../components/Drawer';

const film = {
  title: "Inception",
  synopsis: "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.",
  rating: 4.5,
  posterUrl: "https://example.com/inception-poster.jpg", // Replace with a valid image URL
};

function FilmPage() {
  const { id } = useParams();
  const { data : film, error, loading } = useFetchData(`/movie/${id}`);

  if(loading) {
    return <LoadingInPage />;
  }

  const moviePoster = "https://image.tmdb.org/t/p/w500";

  return (
    <Container maxWidth="sm">
      <SwipeableTemporaryDrawer />
    </Container>
  );
}

export default FilmPage;
