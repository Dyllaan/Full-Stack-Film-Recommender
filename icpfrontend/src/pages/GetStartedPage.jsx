import LoadingInPage from "../components/LoadingInPage";
import useFetchData from "../hooks/useFetchData";
import { useEffect, useState } from "react";
import RateFilmCard from "../components/films/RateFilmCard";
import Pagination from '@mui/material/Pagination';
import RatedFilms from "../components/films/getting-started/RatedFilms";
import Search from "../components/search/Search";
import { Button } from "@mui/material";
/**
 * 
 * Gets the user started to mitigate cold start p roblem
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function GetStartedPage() {
  const endpoint = 'movies';
  const { loading, data, setEndpoint, reloadData } = useFetchData(endpoint);
  const [films, setFilms] = useState([]);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [ratedFilms, setRatedFilms] = useState([]);
  const requiredFilmsToRate = 5;
  

  function handleContinue() {
    if(ratedFilms.length >= requiredFilmsToRate) {
      return <Button className="w-fit" variant="contained" color="primary" href="/">Continue</Button>
    } else {
      return <h2>Rate {requiredFilmsToRate - ratedFilms.length} more films to continue</h2>
    }
  }

  function addRatedFilm(film) {
    setRatedFilms([...ratedFilms, film]);
  }

  useEffect(() => {
    if(data.results) {
      setFilms(data.results);
    }
  }, [data]);

  const handleSearchChange = (e) => {
    setSearch(e.target.value);
  }

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
    }, 300);
    return () => {
      clearTimeout(timer);
    };
  }, [search]);

  useEffect(() => {
    let newEndpoint = `${endpoint}?page=${page}`;

    if (debouncedSearch.trim() !== '') {
      setPage(1);
      newEndpoint += `&search=${debouncedSearch}`;
    }
  
    setEndpoint(newEndpoint);
    reloadData();
  }, [page, debouncedSearch]);

  return (
    <div className="flex flex-col text-center w-full lg:w-[70vw] mx-auto h-[80vh] overflow-hidden">
      <h1 className="text-4xl font-bold">Lets get you started.</h1>
      <div className="p-2">
        {handleContinue()}
      </div>
      <p className="text-xl mt-4">Here are some movies to get you started.</p>
      <div className="p-2">
        <Search placeHolder="Search for a film" search={search} handleSearchChange={handleSearchChange} />
      </div>
      {ratedFilms.length > 0 && <RatedFilms ratedFilms={ratedFilms} />}
      <div className="overflow-y-scroll flex flex-col p-2">
        {loading && <LoadingInPage />}
        <div className="grid sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {!loading && films && films.map((film, index) => (
            <RateFilmCard key={index} film={film} addRatedFilm={addRatedFilm} />
          ))}
        </div>
        <Pagination
          className="mx-auto"
          variant="outlined" 
          bgcolor="secondary" 
          color="primary" 
          count={data.total_pages} 
          page={page} 
          onChange={(event, value) => setPage(value)}
          sx={{
            '.MuiPaginationItem-root': {
              color: '#fff',
          },
      }}
      />
      </div>
    </div>
  );
}

export default GetStartedPage;
