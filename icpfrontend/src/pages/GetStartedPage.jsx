import LoadingInPage from "../components/LoadingInPage";
import useFetchData from "../hooks/useFetchData";
import MoviesHeader from "../components/starter/MoviesHeader";
import { useEffect, useState } from "react";
import PreviewThumbnail from "../components/preview/PreviewThumbnail";
import Pagination from '@mui/material/Pagination';
/**
 * 
 * Gets the user started to mitigate cold start problem
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function GetStartedPage() {
  const endpoint = 'movies';
  const { loading, data, setEndpoint, reloadData } = useFetchData(endpoint);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [debouncedSearch, setDebouncedSearch] = useState("");

  const nextPage = () => {
    setPage(page + 1);
  }

  const backPage = () => {
    if(page > 1) {
      setPage(page - 1);
    }
  }

  const handleSearchChange = (e) => {
    setSearch(e.target.value);
  }

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
    }, 800);
    return () => {
      clearTimeout(timer);
    };
  }, [search]);

  useEffect(() => {
    let newEndpoint = `${endpoint}?page=${page}`;

    if (debouncedSearch.trim() !== '') {
      newEndpoint += `&search=${debouncedSearch}`;
    }
  
    setEndpoint(newEndpoint);
    reloadData();
  }, [page, debouncedSearch]);

  return (
    <div className="flex flex-col text-center w-full lg:w-[70vw] mx-auto">
      <h1 className="text-4xl font-bold">Lets get you started.</h1>
      <p className="text-xl mt-4">Here are some movies to get you started.</p>
      <MoviesHeader page={page} nextPage={nextPage} backPage={backPage} handleSearchChange={handleSearchChange} />
      <div className="mx-auto p-2">
      <Pagination 
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
      {loading && <LoadingInPage />}
      <div className="grid grid-cols-4 gap-4">
        {!loading && data.results && data.results.map((film, index) => (
          <PreviewThumbnail key={index} title={film.movie_title} imgName={film.poster_path} />
        ))}
      </div>
    </div>
  );
}

export default GetStartedPage;
